import os
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class MultiViewDataset(Dataset):
    def __init__(self, data_root: str, split: str = "train", num_views: int = 3, img_size: int = 512, load_masks: bool = False):
        self.data_root = data_root
        self.split = split
        self.num_views = num_views
        self.img_size = img_size
        self.load_masks = load_masks
        self.samples = self._load_sample_list()

    def _load_sample_list(self) -> List[Dict]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        from PIL import Image
        import torchvision.transforms as T

        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        # Center crop to square preserving aspect ratio (matching AnySplat's process_image)
        if width > height:
            new_height = self.img_size
            new_width = int(width * (new_height / height))
        else:
            new_width = self.img_size
            new_height = int(height * (new_width / width))
        img = img.resize((new_width, new_height), Image.BILINEAR)
        left = (new_width - self.img_size) // 2
        top = (new_height - self.img_size) // 2
        img = img.crop((left, top, left + self.img_size, top + self.img_size))
        return T.ToTensor()(img)


class MockMultiViewDataset(Dataset):
    def __init__(self, num_samples: int = 100, num_views: int = 3, img_size: int = 512, num_classes: int = 1):
        self.num_samples = num_samples
        self.num_views = num_views
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        images = torch.randn(self.num_views, 3, self.img_size, self.img_size)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1)
        intrinsics[:, 0, 0] = 500.0 / self.img_size
        intrinsics[:, 1, 1] = 500.0 / self.img_size
        intrinsics[:, 0, 2] = 0.5
        intrinsics[:, 1, 2] = 0.5
        extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1)
        return {"images": images, "intrinsics": intrinsics, "extrinsics": extrinsics, "prompts": None}


class COLMAPDataset(MultiViewDataset):
    def _load_sample_list(self) -> List[Dict]:
        samples = []
        images_dir = os.path.join(self.data_root, "images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        image_files = sorted([f for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
        for i in range(0, len(image_files), self.num_views):
            if i + self.num_views <= len(image_files):
                samples.append({"image_files": [os.path.join(images_dir, image_files[i + v]) for v in range(self.num_views)]})
        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        images = torch.stack([self._preprocess_image(img_file) for img_file in sample["image_files"]], dim=0)
        intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1)
        intrinsics[:, 0, 0] = 500.0 / self.img_size
        intrinsics[:, 1, 1] = 500.0 / self.img_size
        intrinsics[:, 0, 2] = 0.5
        intrinsics[:, 1, 2] = 0.5
        extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1)
        return {"images": images, "intrinsics": intrinsics, "extrinsics": extrinsics, "prompts": None}


class SIU3RDataset(MultiViewDataset):
    def __init__(self, data_root: str, split: str = "train", num_views: int = 3, img_size: int = 518, load_masks: bool = False, frame_stride: int = 1, max_samples: int | None = None, refer_pair_path: str | None = None, prompt_mode: str = "text", target_offset_mult: int = 20):
        self.frame_stride = max(int(frame_stride), 1)
        self.max_samples = max_samples if max_samples is None else max(int(max_samples), 0)
        self.refer_pair_path = refer_pair_path
        self.prompt_mode = prompt_mode
        self.target_offset_mult = target_offset_mult
        self.pre_extract_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "data", "pre_extracted"))
        self.refer_annotations = self._load_refer_annotations(data_root, split)
        self.refer_text_index = self._build_refer_text_index(self.refer_annotations)
        super().__init__(data_root, split, num_views, img_size, load_masks)
        # Filter out training samples with no valid text prompt
        if not self.refer_pair_path:
            self.samples = [s for s in self.samples if s.get("text_prompt")]

    def _load_sample_list(self) -> List[Dict]:
        if self.refer_pair_path:
            return self._load_refer_pair_sample_list()
        samples = []
        split_dir = os.path.join(self.data_root, self.split)
        if not os.path.exists(split_dir):
            print(f"WARNING: dataset directory not found: {split_dir}")
            return []
        for scene_dir in sorted(d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))):
            scene_path = os.path.join(split_dir, scene_dir)
            images_dir = None
            for folder_name in ["images", "color", "imgs"]:
                test_path = os.path.join(scene_path, folder_name)
                if os.path.exists(test_path):
                    images_dir = test_path
                    break
            if images_dir is None:
                continue
            intrinsics_path = os.path.join(scene_path, "intrinsics.npy") if os.path.exists(os.path.join(scene_path, "intrinsics.npy")) else (os.path.join(scene_path, "intrinsic.txt") if os.path.exists(os.path.join(scene_path, "intrinsic.txt")) else None)
            extrinsics_path = os.path.join(scene_path, "extrinsics.npy") if os.path.exists(os.path.join(scene_path, "extrinsics.npy")) else (os.path.join(scene_path, "extrinsic") if os.path.exists(os.path.join(scene_path, "extrinsic")) else None)
            image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
            target_offset = self.num_views * self.target_offset_mult
            for i in range(0, len(image_files), self.num_views * self.frame_stride):
                if self.max_samples is not None and len(samples) >= self.max_samples:
                    return samples
                t = i + target_offset
                has_target = t + self.num_views <= len(image_files)
                if i + self.num_views <= len(image_files):
                    samples.append({
                        "scene_dir": scene_path,
                        "image_files": image_files[i:i + self.num_views],
                        "target_image_files": image_files[t:t + self.num_views] if has_target else [],
                        "intrinsics_path": intrinsics_path,
                        "extrinsics_path": extrinsics_path,
                        "start_idx": i,
                        "target_start_idx": t if has_target else i,
                        "has_camera_params": intrinsics_path is not None and extrinsics_path is not None,
                        "has_target_views": has_target,
                        "text_prompt": self._select_refer_text_prompt(scene_dir, i) or self._load_text_prompt(scene_path) or None,
                    })
        return samples

    def _find_scene_path(self, scene_id: str) -> str | None:
        for split_name in [self.split, "val", "train", "test"]:
            scene_path = os.path.join(self.data_root, split_name, scene_id)
            if os.path.isdir(scene_path):
                return scene_path
        return None

    def _find_images_dir(self, scene_path: str) -> str | None:
        for folder_name in ["images", "color", "imgs"]:
            test_path = os.path.join(scene_path, folder_name)
            if os.path.exists(test_path):
                return test_path
        return None

    def _frame_id_to_image_path(self, images_dir: str, frame_id: int, image_files: List[str]) -> str | None:
        frame_stem = str(int(frame_id))
        candidates = [
            f"{frame_stem}.jpg",
            f"{frame_stem}.png",
            f"{frame_stem}.jpeg",
            f"{int(frame_id):06d}.jpg",
            f"{int(frame_id):06d}.png",
            f"{int(frame_id):06d}.jpeg",
        ]
        for name in candidates:
            path = os.path.join(images_dir, name)
            if os.path.exists(path):
                return path
        if 0 <= int(frame_id) < len(image_files):
            return image_files[int(frame_id)]
        return None

    def _get_object_name(self, scene_id: str, object_id) -> str | None:
        scene_ann = self.refer_annotations.get(scene_id) if isinstance(self.refer_annotations, dict) else None
        if not isinstance(scene_ann, dict):
            return None
        objects = scene_ann.get("objects")
        if not isinstance(objects, dict):
            return None
        object_ann = objects.get(str(object_id))
        if not isinstance(object_ann, dict):
            return None
        object_name = object_ann.get("object_name")
        return object_name.strip() if isinstance(object_name, str) and object_name.strip() else None

    def _load_refer_pair_sample_list(self) -> List[Dict]:
        import json
        pair_path = self.refer_pair_path
        if pair_path and not os.path.isabs(pair_path):
            pair_path = os.path.join(self.data_root, pair_path)
        if not pair_path or not os.path.exists(pair_path):
            print(f"[SIU3RDataset] refer pair file not found: {self.refer_pair_path}")
            return []
        with open(pair_path, "r", encoding="utf-8") as f:
            refer_pairs = json.load(f)
        if not isinstance(refer_pairs, list):
            print(f"[SIU3RDataset] refer pair file must be a list: {pair_path}")
            return []
        samples = []
        for item in refer_pairs:
            if self.max_samples is not None and len(samples) >= self.max_samples:
                break
            if not isinstance(item, dict):
                continue
            scene_id = item.get("scene_name") or item.get("scan")
            context_ids = item.get("context_views_id") or item.get("context_ids")
            if not isinstance(scene_id, str) or not isinstance(context_ids, list) or not context_ids:
                continue
            scene_path = self._find_scene_path(scene_id)
            if scene_path is None:
                continue
            images_dir = self._find_images_dir(scene_path)
            if images_dir is None:
                continue
            all_image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
            image_files = []
            for frame_id in context_ids[: self.num_views]:
                image_path = self._frame_id_to_image_path(images_dir, int(frame_id), all_image_files)
                if image_path:
                    image_files.append(image_path)
            if not image_files:
                continue
            while len(image_files) < self.num_views:
                image_files.append(image_files[-1])
            object_id = item.get("context_objects")
            object_name = self._get_object_name(scene_id, object_id)
            text_prompt = object_name if self.prompt_mode == "object_name" and object_name else item.get("texts")
            if isinstance(text_prompt, list):
                text_prompt = text_prompt[0] if text_prompt else None
            if not isinstance(text_prompt, str) or not text_prompt.strip():
                text_prompt = object_name
            intrinsics_path = os.path.join(scene_path, "intrinsics.npy") if os.path.exists(os.path.join(scene_path, "intrinsics.npy")) else (os.path.join(scene_path, "intrinsic.txt") if os.path.exists(os.path.join(scene_path, "intrinsic.txt")) else None)
            extrinsics_path = os.path.join(scene_path, "extrinsics.npy") if os.path.exists(os.path.join(scene_path, "extrinsics.npy")) else (os.path.join(scene_path, "extrinsic") if os.path.exists(os.path.join(scene_path, "extrinsic")) else None)
            samples.append({
                "scene_dir": scene_path,
                "image_files": image_files[: self.num_views],
                "intrinsics_path": intrinsics_path,
                "extrinsics_path": extrinsics_path,
                "start_idx": int(context_ids[0]),
                "frame_ids": [int(x) for x in context_ids[: self.num_views]],
                "has_camera_params": intrinsics_path is not None and extrinsics_path is not None,
                "text_prompt": text_prompt,
                "object_id": object_id,
                "object_name": object_name,
            })
        print(f"[SIU3RDataset] loaded refer pair samples from {pair_path}: samples={len(samples)}, prompt_mode={self.prompt_mode}")
        return samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        try:
            images = []
            for img_file in sample["image_files"]:
                images.append(self._preprocess_image(img_file) if os.path.exists(img_file) else torch.randn(3, self.img_size, self.img_size))
            images = torch.stack(images, dim=0)
            if sample["has_camera_params"]:
                intrinsics = self._load_intrinsics_from_path(sample["intrinsics_path"])
                extrinsics = self._load_extrinsics_from_path(sample["extrinsics_path"])
                start_idx = sample["start_idx"]
                frame_ids = sample.get("frame_ids")
                if intrinsics.dim() == 3:
                    if isinstance(frame_ids, list) and all(0 <= int(x) < intrinsics.shape[0] for x in frame_ids):
                        intrinsics = intrinsics[torch.tensor(frame_ids, dtype=torch.long)]
                    else:
                        intrinsics = intrinsics[start_idx:start_idx + self.num_views]
                elif intrinsics.dim() == 2:
                    intrinsics = intrinsics.unsqueeze(0).repeat(self.num_views, 1, 1)
                if extrinsics.dim() == 3:
                    if isinstance(frame_ids, list) and all(0 <= int(x) < extrinsics.shape[0] for x in frame_ids):
                        extrinsics = extrinsics[torch.tensor(frame_ids, dtype=torch.long)]
                    else:
                        extrinsics = extrinsics[start_idx:start_idx + self.num_views]
                elif extrinsics.dim() == 2:
                    extrinsics = extrinsics.unsqueeze(0).repeat(self.num_views, 1, 1)
                intrinsics = self._ensure_num_views(intrinsics, 3)
                extrinsics = self._ensure_num_views(extrinsics, 4)
                intrinsics = intrinsics.float()
                extrinsics = extrinsics.float()
                if intrinsics[:, 0, 0].abs().max() > 10.0:
                    intrinsics[:, 0, :] /= self.img_size
                    intrinsics[:, 1, :] /= self.img_size
                intrinsics[:, 0, 2] = 0.5
                intrinsics[:, 1, 2] = 0.5
            else:
                intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1)
                intrinsics[:, 0, 0] = 500.0 / self.img_size
                intrinsics[:, 1, 1] = 500.0 / self.img_size
                intrinsics[:, 0, 2] = 0.5
                intrinsics[:, 1, 2] = 0.5
                extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1)
            prompts = {"text": sample["text_prompt"]} if sample.get("text_prompt") else None
            scene_id = os.path.basename(sample.get("scene_dir", ""))
            start_idx = sample.get("start_idx", 0)
            result = {
                "images": images,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "prompts": prompts,
                "scene_id": scene_id,
                "start_idx": start_idx,
                "frame_ids": sample.get("frame_ids", list(range(start_idx, start_idx + self.num_views))),
                "object_id": sample.get("object_id"),
                "object_name": sample.get("object_name"),
            }
            # Load target views for novel-view render supervision
            if sample.get("has_target_views") and sample.get("target_image_files"):
                target_images = []
                for img_file in sample["target_image_files"]:
                    target_images.append(self._preprocess_image(img_file) if os.path.exists(img_file) else torch.randn(3, self.img_size, self.img_size))
                result["target_images"] = torch.stack(target_images, dim=0)
                # Load target camera params (different frames = different extrinsics)
                if sample["has_camera_params"]:
                    tgt_start = sample.get("target_start_idx", sample["start_idx"])
                    tgt_intrinsics = self._load_intrinsics_from_path(sample["intrinsics_path"])
                    tgt_extrinsics = self._load_extrinsics_from_path(sample["extrinsics_path"])
                    if tgt_intrinsics.dim() == 3:
                        tgt_intrinsics = tgt_intrinsics[tgt_start:tgt_start + self.num_views]
                    elif tgt_intrinsics.dim() == 2:
                        tgt_intrinsics = tgt_intrinsics.unsqueeze(0).repeat(self.num_views, 1, 1)
                    if tgt_extrinsics.dim() == 3:
                        tgt_extrinsics = tgt_extrinsics[tgt_start:tgt_start + self.num_views]
                    elif tgt_extrinsics.dim() == 2:
                        tgt_extrinsics = tgt_extrinsics.unsqueeze(0).repeat(self.num_views, 1, 1)
                    tgt_intrinsics = self._ensure_num_views(tgt_intrinsics, 3).float()
                    tgt_extrinsics = self._ensure_num_views(tgt_extrinsics, 4).float()
                    if tgt_intrinsics[:, 0, 0].abs().max() > 1.0:
                        tgt_intrinsics[:, 0, :] /= self.img_size
                        tgt_intrinsics[:, 1, :] /= self.img_size
                else:
                    tgt_intrinsics = intrinsics
                    tgt_extrinsics = extrinsics
                result["target_intrinsics"] = tgt_intrinsics
                result["target_extrinsics"] = tgt_extrinsics
            pre_extract_path = os.path.join(self.pre_extract_dir, f"{scene_id}_{start_idx}_sam3_qc.pt")
            if os.path.exists(pre_extract_path):
                pre_data = torch.load(pre_extract_path, map_location="cpu", weights_only=True)
                result["sam3_query_class_logits"] = pre_data["query_class_logits"]
                result["sam3_seg_masks"] = pre_data.get("seg_masks")
                result["sam3_query_scores"] = pre_data.get("query_scores")
            # Target view pre-extracted SAM3
            target_start = sample.get("target_start_idx", start_idx)
            result["target_start_idx"] = target_start
            target_feat_path = os.path.join(self.pre_extract_dir, f"{scene_id}_{target_start}_sam3_qc.pt")
            if os.path.exists(target_feat_path) and target_start != start_idx:
                target_pre = torch.load(target_feat_path, map_location="cpu", weights_only=True)
                result["target_sam3_query_class_logits"] = target_pre["query_class_logits"]
            return result
        except Exception:
            images = torch.randn(self.num_views, 3, self.img_size, self.img_size)
            intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1)
            intrinsics[:, 0, 0] = 500.0 / self.img_size
            intrinsics[:, 1, 1] = 500.0 / self.img_size
            intrinsics[:, 0, 2] = 0.5
            intrinsics[:, 1, 2] = 0.5
            extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1)
            return {"images": images, "intrinsics": intrinsics, "extrinsics": extrinsics, "prompts": None}

    def _load_refer_annotations(self, data_root: str, split: str) -> Dict:
        import json
        candidate_files = [
            os.path.join(data_root, f"{split}_refer_seg_data.json"),
            os.path.join(data_root, "val_refer_seg_data.json"),
            os.path.join(data_root, "train_refer_seg_data.json"),
            os.path.join(data_root, "refer_seg_data.json"),
        ]
        merged = {}
        for ann_path in candidate_files:
            if not os.path.exists(ann_path):
                continue
            try:
                with open(ann_path, "r", encoding="utf-8") as f:
                    annotations = json.load(f)
                if isinstance(annotations, dict):
                    new_scenes = set(annotations.keys()) - set(merged.keys())
                    merged.update(annotations)
                    print(f"[SIU3RDataset] loaded refer annotations from {ann_path}, scenes={len(annotations)}, new={len(new_scenes)}, total={len(merged)}")
            except Exception as exc:
                print(f"[SIU3RDataset] failed to load refer annotations from {ann_path}: {exc}")
        return merged

    def _is_natural_language_text(self, value: str) -> bool:
        text = value.strip()
        if len(text) < 8:
            return False
        if text.isdigit():
            return False
        return any(ch.isalpha() for ch in text) and (" " in text or "." in text or "," in text)

    def _collect_text_candidates(self, value) -> List[str]:
        texts = []
        if isinstance(value, str):
            if self._is_natural_language_text(value):
                texts.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                texts.extend(self._collect_text_candidates(item))
        elif isinstance(value, dict):
            for item in value.values():
                texts.extend(self._collect_text_candidates(item))
        return texts

    def _build_refer_text_index(self, annotations: Dict) -> Dict:
        if not isinstance(annotations, dict) or not annotations:
            return {}
        text_index = {}
        for scene_id, scene_ann in annotations.items():
            if not isinstance(scene_ann, dict):
                continue
            frame2object = scene_ann.get("frame2object") if isinstance(scene_ann.get("frame2object"), dict) else {}
            object_texts = {}
            for value in scene_ann.values():
                if not isinstance(value, dict):
                    continue
                for object_id, object_value in value.items():
                    texts = self._collect_text_candidates(object_value)
                    if texts:
                        object_texts.setdefault(str(object_id), []).extend(texts)
            unique_object_texts = {}
            all_texts = []
            for object_id, texts in object_texts.items():
                unique_texts = list(dict.fromkeys(texts))
                unique_object_texts[object_id] = unique_texts
                all_texts.extend(unique_texts)
            all_texts = list(dict.fromkeys(all_texts))
            if not all_texts:
                all_texts = list(dict.fromkeys(self._collect_text_candidates(scene_ann)))
            text_index[scene_id] = {
                "frame2object": frame2object,
                "object_texts": unique_object_texts,
                "all_texts": all_texts,
            }
        indexed_texts = sum(len(scene_data.get("all_texts", [])) for scene_data in text_index.values())
        print(f"[SIU3RDataset] built refer text index: scenes={len(text_index)}, texts={indexed_texts}")
        return text_index

    def _select_refer_text_prompt(self, scene_id: str, start_idx: int) -> str | None:
        scene_data = self.refer_text_index.get(scene_id)
        if not isinstance(scene_data, dict):
            return None
        visible_object_ids = []
        frame2object = scene_data.get("frame2object")
        if isinstance(frame2object, dict):
            for frame_idx in range(start_idx, start_idx + self.num_views):
                visible_object_ids.extend(str(x) for x in frame2object.get(str(frame_idx), []))
        visible_object_ids = list(dict.fromkeys(visible_object_ids))
        candidate_texts = []
        object_texts = scene_data.get("object_texts", {})
        if isinstance(object_texts, dict):
            for object_id in visible_object_ids:
                candidate_texts.extend(object_texts.get(object_id, []))
        if not candidate_texts:
            candidate_texts = scene_data.get("all_texts", [])
        unique_texts = list(dict.fromkeys(text for text in candidate_texts if text))
        if not unique_texts:
            return None
        return unique_texts[start_idx % len(unique_texts)]

    def _load_text_prompt(self, scene_path: str) -> str | None:
        candidate_files = [
            "text.txt",
            "prompt.txt",
            "caption.txt",
            "captions.txt",
            "description.txt",
            "label.txt",
            "labels.txt",
            "class.txt",
            "classes.txt",
        ]
        for filename in candidate_files:
            text_path = os.path.join(scene_path, filename)
            if not os.path.exists(text_path):
                continue
            try:
                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()
                if text:
                    return text
            except Exception:
                continue
        return os.path.basename(scene_path) if scene_path else None

    def _ensure_num_views(self, tensor: torch.Tensor, matrix_size: int) -> torch.Tensor:
        if tensor.dim() != 3 or tensor.shape[-2:] != (matrix_size, matrix_size):
            return torch.eye(matrix_size).unsqueeze(0).repeat(self.num_views, 1, 1)
        if tensor.shape[0] >= self.num_views:
            return tensor[: self.num_views]
        pad = tensor[-1:].repeat(self.num_views - tensor.shape[0], 1, 1) if tensor.shape[0] > 0 else torch.eye(matrix_size).unsqueeze(0).repeat(self.num_views, 1, 1)
        return torch.cat([tensor, pad], dim=0) if tensor.shape[0] > 0 else pad

    def _load_intrinsics_from_path(self, path: str) -> torch.Tensor:
        import numpy as np

        def _to_intrinsic_tensor(array) -> torch.Tensor:
            arr = np.asarray(array, dtype=np.float32)
            if arr.shape == (3, 3):
                return torch.from_numpy(arr)
            if arr.shape == (4, 4):
                return torch.from_numpy(arr[:3, :3])
            if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
                return torch.from_numpy(arr)
            if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                return torch.from_numpy(arr[:, :3, :3])
            if arr.ndim == 2 and arr.shape[1] == 9:
                return torch.from_numpy(arr.reshape(-1, 3, 3))
            if arr.ndim == 1 and arr.size == 9:
                return torch.from_numpy(arr.reshape(3, 3))
            if arr.ndim == 2 and arr.shape[1] == 4:
                mats = np.tile(np.eye(3, dtype=np.float32), (arr.shape[0], 1, 1))
                mats[:, 0, 0] = arr[:, 0]
                mats[:, 1, 1] = arr[:, 1]
                mats[:, 0, 2] = arr[:, 2]
                mats[:, 1, 2] = arr[:, 3]
                return torch.from_numpy(mats)
            if arr.ndim == 1 and arr.size == 4:
                mat = np.eye(3, dtype=np.float32)
                mat[0, 0], mat[1, 1], mat[0, 2], mat[1, 2] = arr.tolist()
                return torch.from_numpy(mat)
            raise ValueError(f"Unsupported intrinsic format: shape={arr.shape}")

        raw = np.load(path) if path.endswith(".npy") else np.loadtxt(path)
        try:
            parsed = _to_intrinsic_tensor(raw)
            print(f"[SIU3RDataset] loaded intrinsics from {path} with raw_shape={np.asarray(raw).shape} -> parsed_shape={tuple(parsed.shape)}")
            return parsed
        except Exception as exc:
            print(f"[SIU3RDataset] failed to parse intrinsics from {path} with raw_shape={np.asarray(raw).shape}: {exc}; using identity intrinsics fallback.")
            return torch.eye(3)

    def _load_extrinsics_from_path(self, path: str) -> torch.Tensor:
        import numpy as np
        if os.path.isdir(path):
            all_extrinsics = []
            for f in sorted(os.path.join(path, x) for x in os.listdir(path) if x.endswith((".txt", ".npy"))):
                ext = np.loadtxt(f) if f.endswith(".txt") else np.load(f)
                if ext.shape == (4, 4):
                    all_extrinsics.append(ext)
            return torch.from_numpy(np.stack(all_extrinsics, axis=0)) if all_extrinsics else torch.eye(4).unsqueeze(0)
        if path.endswith(".npy"):
            return torch.from_numpy(np.load(path))
        if path.endswith(".txt"):
            extrinsics = np.loadtxt(path)
            return torch.from_numpy(extrinsics) if extrinsics.shape == (4, 4) else torch.eye(4)
        return torch.eye(4)


def create_dataloader(dataset: Dataset, batch_size: int = 2, num_workers: int = 4, shuffle: bool = True, pin_memory: bool = True, persistent_workers: bool = True, prefetch_factor: int = 2, drop_last: bool = True) -> torch.utils.data.DataLoader:
    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            num_views = getattr(dataset, "num_views", 3)
            img_size = getattr(dataset, "img_size", 518)
            return {
                "images": torch.randn(batch_size, num_views, 3, img_size, img_size),
                "intrinsics": torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1),
                "extrinsics": torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1),
                "prompts": None,
            }
        result = {}
        skip_stack = {"sam3_query_class_logits", "sam3_seg_masks", "sam3_query_scores"}
        for key in batch[0].keys():
            values = [b[key] for b in batch]
            if values[0] is None:
                result[key] = None
            elif key in skip_stack:
                result[key] = values[0] if len(values) == 1 else values
            elif isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = values
        return result

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    return torch.utils.data.DataLoader(**loader_kwargs)
