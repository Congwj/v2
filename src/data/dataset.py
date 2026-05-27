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
    def __init__(self, data_root: str, split: str = "train", num_views: int = 3, img_size: int = 518, load_masks: bool = False, max_samples: int | None = None, refer_pair_path: str | None = None, prompt_mode: str = "text", min_view_gap: int = 3):
        self.max_samples = max_samples if max_samples is None else max(int(max_samples), 0)
        self.refer_pair_path = refer_pair_path
        self.prompt_mode = prompt_mode
        self.min_view_gap = max(int(min_view_gap), 1)
        self.pre_extract_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "data", "pre_extracted"))
        self.refer_annotations = self._load_refer_annotations(data_root, split)
        self.refer_text_index = self._build_refer_text_index(self.refer_annotations)
        super().__init__(data_root, split, num_views, img_size, load_masks)
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
            image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith((".jpg", ".png", ".jpeg"))])
            n_frames = len(image_files)
            if n_frames < self.num_views * 2:
                continue
            intrinsics_path = os.path.join(scene_path, "intrinsics.npy") if os.path.exists(os.path.join(scene_path, "intrinsics.npy")) else (os.path.join(scene_path, "intrinsic.txt") if os.path.exists(os.path.join(scene_path, "intrinsic.txt")) else None)
            extrinsics_path = os.path.join(scene_path, "extrinsics.npy") if os.path.exists(os.path.join(scene_path, "extrinsics.npy")) else (os.path.join(scene_path, "extrinsic") if os.path.exists(os.path.join(scene_path, "extrinsic")) else None)
            # Each scene contributes multiple sampling opportunities
            n_samples = max(1, min(n_frames // (self.num_views * 2), 200))
            for _ in range(n_samples):
                if self.max_samples is not None and len(samples) >= self.max_samples:
                    return samples
                samples.append({
                    "scene_dir": scene_path,
                    "n_frames": n_frames,
                    "image_files": image_files,
                    "intrinsics_path": intrinsics_path,
                    "extrinsics_path": extrinsics_path,
                    "has_camera_params": intrinsics_path is not None and extrinsics_path is not None,
                    "text_prompt": self._select_refer_text_prompt(scene_dir, 0) or self._load_text_prompt(scene_path) or None,
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

    def _sample_views(self, pool_size: int):
        """Randomly sample source and target view indices from pool [0, pool_size) with minimum gap."""
        import random
        all_frames = list(range(pool_size))
        random.shuffle(all_frames)
        src_idx = []
        tgt_idx = []
        for f in all_frames:
            if len(src_idx) < self.num_views:
                if all(abs(f - s) >= self.min_view_gap for s in src_idx):
                    src_idx.append(f)
            elif len(tgt_idx) < self.num_views:
                if all(abs(f - t) >= self.min_view_gap for t in tgt_idx) and all(abs(f - s) >= self.min_view_gap for s in src_idx):
                    tgt_idx.append(f)
            if len(src_idx) >= self.num_views and len(tgt_idx) >= self.num_views:
                break
        # Fallback
        remaining = [f for f in all_frames if f not in src_idx and f not in tgt_idx]
        if len(src_idx) < self.num_views:
            src_idx.extend(remaining[:self.num_views - len(src_idx)])
        if len(tgt_idx) < self.num_views:
            remaining2 = [f for f in remaining if f not in src_idx]
            tgt_idx.extend(remaining2[:self.num_views - len(tgt_idx)])
        return sorted(src_idx), sorted(tgt_idx)

    def _load_cameras_for_frames(self, intrinsics_path, extrinsics_path, frame_ids):
        """Load camera params for specific frame indices."""
        intrinsics = self._load_intrinsics_from_path(intrinsics_path)
        extrinsics = self._load_extrinsics_from_path(extrinsics_path)
        if intrinsics.dim() == 3:
            if all(0 <= int(x) < intrinsics.shape[0] for x in frame_ids):
                intrinsics = intrinsics[torch.tensor(frame_ids, dtype=torch.long)]
            else:
                intrinsics = intrinsics[frame_ids[0]:frame_ids[0] + len(frame_ids)]
        elif intrinsics.dim() == 2:
            intrinsics = intrinsics.unsqueeze(0).repeat(len(frame_ids), 1, 1)
        if extrinsics.dim() == 3:
            if all(0 <= int(x) < extrinsics.shape[0] for x in frame_ids):
                extrinsics = extrinsics[torch.tensor(frame_ids, dtype=torch.long)]
            else:
                extrinsics = extrinsics[frame_ids[0]:frame_ids[0] + len(frame_ids)]
        elif extrinsics.dim() == 2:
            extrinsics = extrinsics.unsqueeze(0).repeat(len(frame_ids), 1, 1)
        intrinsics = self._ensure_num_views(intrinsics, 3).float()
        extrinsics = self._ensure_num_views(extrinsics, 4).float()
        if intrinsics[:, 0, 0].abs().max() > 10.0:
            intrinsics[:, 0, :] /= self.img_size
            intrinsics[:, 1, :] /= self.img_size
        intrinsics[:, 0, 2] = 0.5
        intrinsics[:, 1, 2] = 0.5
        return intrinsics, extrinsics

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        try:
            scene_id = os.path.basename(sample.get("scene_dir", ""))
            result = {}

            # Two sample formats: pool-based (train) vs refer-pair (inference)
            is_pool = "n_frames" in sample

            if is_pool:
                image_files = sample["image_files"]
                pool_path = os.path.join(self.pre_extract_dir, f"{scene_id}_sam3.pt")
                if os.path.exists(pool_path):
                    pool_data = torch.load(pool_path, map_location="cpu", weights_only=True)
                    frame_indices = pool_data.get("frame_indices", list(range(pool_data["query_class_logits"].shape[1])))
                    pool_size = len(frame_indices)
                    src_rel, tgt_rel = self._sample_views(pool_size)
                    src_idx = [frame_indices[r] for r in src_rel]
                    tgt_idx = [frame_indices[r] for r in tgt_rel]
                    images = torch.stack([self._preprocess_image(image_files[i]) for i in src_idx], dim=0)
                    target_images = torch.stack([self._preprocess_image(image_files[i]) for i in tgt_idx], dim=0)
                    all_qc = pool_data["query_class_logits"]  # [1, 20, Q, C, H, W]
                    result["sam3_query_class_logits"] = all_qc[:, src_rel, ...]
                    result["target_sam3_query_class_logits"] = all_qc[:, tgt_rel, ...]
                else:
                    raise FileNotFoundError(f"Pool file not found: {pool_path}. Run scripts/pre_extract_sam3.py first.")
            else:
                # Refer-pair format: fixed image_files list
                image_files = sample["image_files"]
                src_idx = list(range(len(image_files)))
                tgt_idx = list(range(len(image_files)))
                images = torch.stack([self._preprocess_image(f) for f in image_files], dim=0)
                target_images = images.clone()

            # Load cameras
            if sample.get("has_camera_params"):
                intrinsics, extrinsics = self._load_cameras_for_frames(
                    sample["intrinsics_path"], sample["extrinsics_path"], src_idx)
                tgt_intrinsics, tgt_extrinsics = self._load_cameras_for_frames(
                    sample["intrinsics_path"], sample["extrinsics_path"], tgt_idx)
            else:
                intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1)
                intrinsics[:, 0, 0] = 500.0 / self.img_size; intrinsics[:, 1, 1] = 500.0 / self.img_size
                intrinsics[:, 0, 2] = 0.5; intrinsics[:, 1, 2] = 0.5
                extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1)
                tgt_intrinsics = intrinsics; tgt_extrinsics = extrinsics

            prompts = {"text": sample["text_prompt"]} if sample.get("text_prompt") else None
            result.update({
                "images": images,
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "prompts": prompts,
                "scene_id": scene_id,
                "start_idx": src_idx[0] if src_idx else 0,
                "frame_ids": src_idx,
                "target_images": target_images,
                "target_intrinsics": tgt_intrinsics,
                "target_extrinsics": tgt_extrinsics,
                "target_start_idx": tgt_idx[0] if tgt_idx else 0,
            })
            return result
        except Exception as e:
            print(f"[SIU3RDataset] ERROR loading sample {idx}: {e}")
            images = torch.randn(self.num_views, 3, self.img_size, self.img_size)
            intrinsics = torch.eye(3).unsqueeze(0).repeat(self.num_views, 1, 1)
            intrinsics[:, 0, 0] = 500.0 / self.img_size; intrinsics[:, 1, 1] = 500.0 / self.img_size
            intrinsics[:, 0, 2] = 0.5; intrinsics[:, 1, 2] = 0.5
            extrinsics = torch.eye(4).unsqueeze(0).repeat(self.num_views, 1, 1)
            return {"images": images, "intrinsics": intrinsics, "extrinsics": extrinsics, "prompts": None,
                    "scene_id": "", "start_idx": 0, "target_start_idx": 0, "frame_ids": [],
                    "target_images": images, "target_intrinsics": intrinsics, "target_extrinsics": extrinsics}

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
        skip_stack = {"sam3_query_class_logits", "target_sam3_query_class_logits", "sam3_seg_masks", "sam3_query_scores"}
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
