import os, json, requests, random, time, runpod

import os
import cv2
from tqdm import tqdm
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils import img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from realesrgan.utils import RealESRGANer
from lightning_models.mmse_rectified_flow import MMSERectifiedFlow

device = "cuda"

def set_realesrgan():
    use_half = False
    if torch.cuda.is_available():
        no_half_gpu_list = ["1650", "1660"]
        if not True in [
            gpu in torch.cuda.get_device_name(0) for gpu in no_half_gpu_list
        ]:
            use_half = True

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="/content/PMRF/pretrained_models/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=use_half,
    )
    return upsampler

upsampler = set_realesrgan()
pmrf = MMSERectifiedFlow.from_pretrained("/content/PMRF/model").to(device=device)

def generate_reconstructions(pmrf_model, x, y, non_noisy_z0, num_flow_steps, device):
    source_dist_samples = pmrf_model.create_source_distribution_samples(
        x, y, non_noisy_z0
    )
    dt = (1.0 / num_flow_steps) * (1.0 - pmrf_model.hparams.eps)
    x_t_next = source_dist_samples.clone()
    t_one = torch.ones(x.shape[0], device=device)
    for i in tqdm(range(num_flow_steps)):
        num_t = (i / num_flow_steps) * (
            1.0 - pmrf_model.hparams.eps
        ) + pmrf_model.hparams.eps
        v_t_next = pmrf_model(x_t=x_t_next, t=t_one * num_t, y=y).to(x_t_next.dtype)
        x_t_next = x_t_next.clone() + v_t_next * dt

    return x_t_next.clip(0, 1)

def resize(img, size):
    # From https://github.com/sczhou/CodeFormer/blob/master/facelib/utils/face_restoration_helper.py
    h, w = img.shape[0:2]
    scale = size / min(h, w)
    h, w = int(h * scale), int(w * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (w, h), interpolation=interp)

@torch.inference_mode()
def enhance_face(img, face_helper, has_aligned, num_flow_steps, scale=2):
    face_helper.clean_all()
    if has_aligned:  # The inputs are already aligned
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        face_helper.cropped_faces = [img]
    else:
        face_helper.read_image(img)
        face_helper.input_img = resize(face_helper.input_img, 640)
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        face_helper.align_warp_face()
    if len(face_helper.cropped_faces) == 0:
        raise Exception("Could not identify any face in the image.")
    if has_aligned and len(face_helper.cropped_faces) > 1:
        raise Exception(
            "You marked that the input image is aligned, but multiple faces were detected."
        )

    # face restoration
    for i, cropped_face in tqdm(enumerate(face_helper.cropped_faces)):
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        output = generate_reconstructions(
            pmrf,
            torch.zeros_like(cropped_face_t),
            cropped_face_t,
            None,
            num_flow_steps,
            device,
        )
        restored_face = tensor2img(
            output.to(torch.float32).squeeze(0), rgb2bgr=True, min_max=(0, 1)
        )
        restored_face = restored_face.astype("uint8")
        face_helper.add_restored_face(restored_face)

    if not has_aligned:
        # upsample the background
        # Now only support RealESRGAN for upsampling background
        bg_img = upsampler.enhance(img, outscale=scale)[0]
        face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = face_helper.paste_faces_to_input_image(upsample_img=bg_img)
        return face_helper.cropped_faces, face_helper.restored_faces, restored_img
    else:
        return face_helper.cropped_faces, face_helper.restored_faces, None

@torch.inference_mode()
def inference(img_path, randomize_seed, aligned, scale, num_flow_steps, seed):
    if img_path is None:
        raise Exception("Please upload an image before submitting.")
    if randomize_seed:
        random.seed(int(time.time()))
        seed = random.randint(0, 18446744073709551615)
    torch.manual_seed(seed)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w = img.shape[0:2]
    if h > 4500 or w > 4500:
        raise Exception("Image size too large.")

    face_helper = FaceRestoreHelper(
        scale,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
        model_rootpath=None,
    )

    has_aligned = aligned
    cropped_face, restored_faces, restored_img = enhance_face(
        img, face_helper, has_aligned, num_flow_steps=num_flow_steps, scale=scale
    )
    if has_aligned:
        output = restored_faces[0]
    else:
        output = restored_img

    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    for i, restored_face in enumerate(restored_faces):
        restored_faces[i] = cv2.cvtColor(restored_face, cv2.COLOR_BGR2RGB)
    torch.cuda.empty_cache()
    return output, restored_faces if len(restored_faces) > 1 else None

def download_file(url, save_dir, file_name):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    input_image=values['input_image_check']
    input_image=download_file(url=input_image, save_dir='/content/ComfyUI/input', file_name='input_image')
    aligned = values['aligned']
    scale = values['scale']
    num_flow_steps = values['num_flow_steps']
    randomize_seed = values['randomize_seed']
    seed = values['seed']

    output_image, restored_faces = inference(input_image, randomize_seed=randomize_seed, aligned=aligned, scale=scale, num_flow_steps=num_flow_steps, seed=seed)
    output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("/content/pmrf-tost.png", output_image_bgr)

    result = '/content/pmrf-tost.png'
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})