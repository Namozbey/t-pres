from depth_anything_3.api import DepthAnything3
import gc
import cv2
import torch
import time
import numpy as np

class DA3:
    def load(self):
        # model = DepthAnything3.from_pretrained("depth-anything/DA3MONO-LARGE")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE-1.1")
        # model = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE")
        model = model.to(device)
        model.eval()
        print(f"Model loaded on {device}")
        
        self.run_times = []
        self.model = model
        self.device = device

    def _cleanup_additional_resources(self):
        """
        Hook method for child classes to clean up additional resources.
        Override this method if your pipeline has additional attributes to clean up
        (e.g., compel, tokenizers, etc.).
        """
        if hasattr(self, "compel") and self.compel is not None:
            self.compel = None

    def unload(self):
        """
        Unload the pipeline from memory.
        Handles common cleanup: deletes the pipe, clears CUDA cache, and runs garbage collection.
        Child classes can override _cleanup_additional_resources() for additional cleanup.
        """
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
        self._cleanup_additional_resources()
        torch.cuda.empty_cache()
        gc.collect()

    def forward(self, img_path):
        start = time.perf_counter()

        original_image = cv2.imread(img_path)
        H_orig, W_orig = original_image.shape[:2]
        longest_edge = max(H_orig, W_orig)
        optimal_res = int(round(longest_edge / (14.0)) * 14)
        process_res = min(optimal_res, 1330)
        print(f"Running native inference at process_res: {process_res}")

        prediction = self.model.inference(image=[img_path], process_res=process_res)

        depth = prediction.depth[0] # Depth in [m].
        print("depth:", depth.shape)

        depth_resized = cv2.resize(
            depth, 
            (W_orig, H_orig), 
            interpolation=cv2.INTER_LINEAR  # Change to cv2.INTER_NEAREST if edges stretch in 3D
        )
        depth_resized = depth_resized.astype(np.float32)
        print("depth_resized:", depth_resized.shape)

        if prediction.intrinsics is None:
            return depth_resized, None, None, None, None, None, 

        H_pred, W_pred = depth.shape

        scale_x = W_orig / W_pred
        scale_y = H_orig / H_pred
        fx = prediction.intrinsics[0, 0, 0] * scale_x
        fy = prediction.intrinsics[0, 1, 1] * scale_y
        cx = prediction.intrinsics[0, 0, 2] * scale_x
        cy = prediction.intrinsics[0, 1, 2] * scale_y

        h, w = depth_resized.shape
        SENSOR_HEIGHT_MM = 24.0  # Standard full-frame sensor height
        focal_length = (fy / h) * SENSOR_HEIGHT_MM
        print("focal_length:", focal_length)

        end = time.perf_counter()
        runtime = round(end - start, 3)
        self.run_times.append(runtime)

        return depth_resized, focal_length, fx, fy, cx, cy

    def mean_runtime (self):
        arr = np.array(self.run_times)
        return arr.mean()