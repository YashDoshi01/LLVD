import cv2
import numpy as np
import json
import os
import time
import psutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from sklearn.cluster import DBSCAN




class VideoReader:
    
    def __init__(self, path):
        self.path = path
        self.is_image_sequence = os.path.isdir(path)
        
        if self.is_image_sequence:
            exts = ('.jpg', '.jpeg', '.png', '.bmp')
            self.image_files = sorted([
                os.path.join(path, f) for f in os.listdir(path)
                if f.lower().endswith(exts)
            ])
            self.total_frames = len(self.image_files)
            if self.total_frames == 0:
                raise ValueError(f"No image files found in directory: {path}")
            self.current_idx = 0
            
            first_frame = cv2.imread(self.image_files[0])
            if first_frame is None:
                raise ValueError(f"Could not read first image: {self.image_files[0]}")
            self.height, self.width = first_frame.shape[:2]
            self.fps = 25.0
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video file: {path}")
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    
    def read(self):
        if self.is_image_sequence:
            if self.current_idx >= len(self.image_files):
                return False, None
            frame = cv2.imread(self.image_files[self.current_idx])
            self.current_idx += 1
            if frame is None:
                return False, None
            return True, frame
        else:
            return self.cap.read()
    
    def release(self):
        if not self.is_image_sequence:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()



def create_grid(roi_config, num_rows, num_cols):
    """Create grid configuration."""
    cell_w = roi_config['w'] // num_cols
    cell_h = roi_config['h'] // num_rows
    return {
        'x': roi_config['x'], 'y': roi_config['y'],
        'w': roi_config['w'], 'h': roi_config['h'],
        'rows': num_rows, 'cols': num_cols,
        'cell_w': cell_w, 'cell_h': cell_h
    }


def process_grid_detection(roi_data, grid_config, min_pixels=100):
    """Process grid-based detection."""
    result = np.zeros((grid_config['rows'], grid_config['cols']), dtype=np.int8)
    
    detection_masks = []
    for channel in roi_data:
        blurred = cv2.GaussianBlur(channel, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        detection_masks.append(dilated)
    
    if len(detection_masks) > 1:
        combined_mask = np.all(np.stack(detection_masks), axis=0).astype(np.uint8) * 255
    else:
        combined_mask = detection_masks[0]
    
    for row in range(grid_config['rows']):
        for col in range(grid_config['cols']):
            y_start = row * grid_config['cell_h']
            y_end = (row + 1) * grid_config['cell_h']
            x_start = col * grid_config['cell_w']
            x_end = (col + 1) * grid_config['cell_w']
            
            cell = combined_mask[y_start:y_end, x_start:x_end]
            if np.sum(cell > 0) >= min_pixels:
                result[row, col] = 1
    
    return result


def cluster_active_cells(result_matrix, grid_config, eps=1.5, min_samples=1):
    """Cluster active cells using DBSCAN."""
    active_coords = []
    for row in range(grid_config['rows']):
        for col in range(grid_config['cols']):
            if result_matrix[row, col] == 1:
                active_coords.append([row, col])
    
    if len(active_coords) == 0:
        return []
    
    active_coords = np.array(active_coords)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(active_coords)
    labels = clustering.labels_
    
    clusters = []
    for label in set(labels):
        if label == -1:
            continue
        cluster_cells = active_coords[labels == label]
        clusters.append(cluster_cells.tolist())
    
    return clusters


def generate_bounding_boxes(clusters, grid_config):
    """Generate bounding boxes from clusters."""
    bboxes = []
    for cluster in clusters:
        if len(cluster) == 0:
            continue
        cluster = np.array(cluster)
        min_row, min_col = cluster.min(axis=0)
        max_row, max_col = cluster.max(axis=0)
        
        x1 = grid_config['x'] + min_col * grid_config['cell_w']
        y1 = grid_config['y'] + min_row * grid_config['cell_h']
        x2 = grid_config['x'] + (max_col + 1) * grid_config['cell_w']
        y2 = grid_config['y'] + (max_row + 1) * grid_config['cell_h']
        
        bboxes.append([x1, y1, x2, y2])
    return bboxes


def visualize_grid(frame, grid_config, result_matrix):
    """Draw grid overlay."""
    for row in range(grid_config['rows']):
        for col in range(grid_config['cols']):
            x = grid_config['x'] + col * grid_config['cell_w']
            y = grid_config['y'] + row * grid_config['cell_h']
            
            if result_matrix[row, col] == 1:
                color, thickness = (0, 255, 0), 2
            else:
                color, thickness = (128, 128, 128), 1
            
            cv2.rectangle(frame, (x, y), 
                         (x + grid_config['cell_w'], y + grid_config['cell_h']),
                         color, thickness)


def draw_bounding_boxes(frame, bboxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes."""
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def compute_density(result_matrix):
    """Compute density."""
    total_cells = result_matrix.size
    active_cells = np.sum(result_matrix)
    return active_cells / total_cells if total_cells > 0 else 0.0


# ====================================================================
# WORKER FUNCTION (runs in separate process)
# ====================================================================

def process_frame_pair_worker(args):
    """
    Worker function to process a single frame pair.
    Must be at module level for Windows multiprocessing.
    """
    (frame_idx, f1, f2, roi1_cfg, roi2_cfg, grid1_cfg, grid2_cfg, channels) = args
    
    if f1 is None or f2 is None or f1.shape[:2] != f2.shape[:2]:
        return None
    
    # Extract ROIs
    roi1_f1 = f1[roi1_cfg['y']:roi1_cfg['y']+roi1_cfg['h'], 
                  roi1_cfg['x']:roi1_cfg['x']+roi1_cfg['w']]
    roi1_f2 = f2[roi1_cfg['y']:roi1_cfg['y']+roi1_cfg['h'], 
                  roi1_cfg['x']:roi1_cfg['x']+roi1_cfg['w']]
    
    roi2_f1 = f1[roi2_cfg['y']:roi2_cfg['y']+roi2_cfg['h'], 
                  roi2_cfg['x']:roi2_cfg['x']+roi2_cfg['w']]
    roi2_f2 = f2[roi2_cfg['y']:roi2_cfg['y']+roi2_cfg['h'], 
                  roi2_cfg['x']:roi2_cfg['x']+roi2_cfg['w']]
    
    # Frame differencing
    diff1 = cv2.absdiff(roi1_f1, roi1_f2)
    diff2 = cv2.absdiff(roi2_f1, roi2_f2)
    
    # Convert to color space
    if channels == 'gray':
        roi1_data = [cv2.cvtColor(diff1, cv2.COLOR_BGR2GRAY)]
        roi2_data = [cv2.cvtColor(diff2, cv2.COLOR_BGR2GRAY)]
    else:
        hsv1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2HSV)
        roi1_data = [hsv1[:, :, ch] for ch in channels]
        roi2_data = [hsv2[:, :, ch] for ch in channels]
    
    # Process grids
    result_matrix1 = process_grid_detection(roi1_data, grid1_cfg)
    result_matrix2 = process_grid_detection(roi2_data, grid2_cfg)
    
    # Clustering
    clusters1 = cluster_active_cells(result_matrix1, grid1_cfg)
    clusters2 = cluster_active_cells(result_matrix2, grid2_cfg)
    
    vehicle_count1 = len(clusters1)
    vehicle_count2 = len(clusters2)
    
    density1 = compute_density(result_matrix1)
    density2 = compute_density(result_matrix2)
    
    # Return only what we need to render & stats
    return {
        'frame_idx': frame_idx,
        'matrix1': result_matrix1,
        'matrix2': result_matrix2,
        'clusters1': clusters1,
        'clusters2': clusters2,
        'vehicle_count1': vehicle_count1,
        'vehicle_count2': vehicle_count2,
        'density1': density1,
        'density2': density2,
        # use the second frame as base for drawing (like original)
        'frame': f2
    }


# ====================================================================
# MAIN PROCESSOR
# ====================================================================

class MultiprocessingTPL:
    def __init__(self, config_path="user_input_data.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.video_path = self.config["video"]
        color_channel = self.config["color_channel"]
        
        choices = {
            'H': [0], 'S': [1], 'V': [2],
            'H+S': [0, 1], 'H+V': [0, 2], 'S+V': [1, 2],
            'H+S+V': [0, 1, 2], 'gray': 'gray'
        }
        self.channels = choices[color_channel]
        
        # Original ROIs
        self.roi1 = {'x': 545, 'y': 159, 'w': 284, 'h': 140}
        self.roi2 = {'x': 238, 'y': 161, 'w': 284, 'h': 140}
        
        rows, cols = self.config["grids"]["rows"], self.config["grids"]["cols"]
        self.grid1 = create_grid(self.roi1, rows, cols)
        self.grid2 = create_grid(self.roi2, rows, cols)
        
        self.density_lane1 = []
        self.density_lane2 = []
        self.vehicle_counts_lane1 = []
        self.vehicle_counts_lane2 = []
        
        self.num_workers = max(1, cpu_count() - 2)  # Leave some cores free
        self.batch_size = int(self.config.get("batch_size", 64))  # configurable batch size
    
    def run(self, output_path="output/videos/base_pipeline_output.avi"):
        """Run multiprocessing version with batched streaming."""
        print("="*70)
        print("V1: Multiprocessing TPL1 + TPL2 (Batched & Streamed)")
        print(f"Workers: {self.num_workers}")
        print(f"Batch size: {self.batch_size}")
        print("="*70)
        
        start_time = time.time()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with VideoReader(self.video_path) as reader:
            print(f"Video: {self.video_path}")
            print(f"Dimensions: {reader.width}x{reader.height}")
            print(f"Total frames: {reader.total_frames}")
            
            if reader.total_frames < 2:
                print("Not enough frames to form pairs. Exiting.")
                return
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                output_path, fourcc, reader.fps,
                (reader.width, reader.height)
            )
            
            # Total number of frame pairs
            total_pairs = max(reader.total_frames - 1, 0)
            
            # Read first frame
            ret, prev_frame = reader.read()
            if not ret or prev_frame is None:
                print("Failed to read first frame.")
                out.release()
                return
            
            frame_idx = 1
            
            # Multiprocessing pool
            with Pool(processes=self.num_workers) as pool:
                pbar = tqdm(total=total_pairs, desc="Processing", unit="pair")
                
                while True:
                    tasks = []
                    
                    # Build one batch of tasks
                    while len(tasks) < self.batch_size and frame_idx < reader.total_frames:
                        ret, curr_frame = reader.read()
                        if not ret or curr_frame is None:
                            break
                        
                        # Copy frames so each worker has its own data
                        f1 = prev_frame.copy()
                        f2 = curr_frame.copy()
                        
                        tasks.append(
                            (frame_idx, f1, f2,
                             self.roi1, self.roi2,
                             self.grid1, self.grid2,
                             self.channels)
                        )
                        
                        prev_frame = curr_frame
                        frame_idx += 1
                    
                    if not tasks:
                        break  # no more data
                    
                    # Process current batch in parallel
                    # map preserves input order -> frames will be in sequence
                    results = pool.map(process_frame_pair_worker, tasks)
                    
                    # Immediately write results and accumulate stats
                    for result in results:
                        if result is None:
                            pbar.update(1)
                            continue
                        
                        frame = result['frame'].copy()
                        
                        # Overlay grids
                        visualize_grid(frame, self.grid1, result['matrix1'])
                        visualize_grid(frame, self.grid2, result['matrix2'])
                        
                        # Bounding boxes
                        bboxes1 = generate_bounding_boxes(result['clusters1'], self.grid1)
                        bboxes2 = generate_bounding_boxes(result['clusters2'], self.grid2)
                        
                        draw_bounding_boxes(frame, bboxes1, (0, 255, 0), 2)
                        draw_bounding_boxes(frame, bboxes2, (0, 255, 255), 2)
                        
                        # Text overlay
                        cv2.putText(
                            frame,
                            f"Frame: {result['frame_idx']}",
                            (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                        cv2.putText(
                            frame,
                            f"Lane1: {result['vehicle_count1']} veh, D: {result['density1']:.3f}",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                        cv2.putText(
                            frame,
                            f"Lane2: {result['vehicle_count2']} veh, D: {result['density2']:.3f}",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2
                        )
                        
                        # Write immediately
                        out.write(frame)
                        
                        # Accumulate stats
                        self.density_lane1.append(result['density1'])
                        self.density_lane2.append(result['density2'])
                        self.vehicle_counts_lane1.append(result['vehicle_count1'])
                        self.vehicle_counts_lane2.append(result['vehicle_count2'])
                        
                        pbar.update(1)
                
                pbar.close()
            
            out.release()
        
        elapsed = time.time() - start_time
        processed_frames = len(self.density_lane1)  # number of frame pairs processed
        fps = processed_frames / elapsed if elapsed > 0 else 0.0
        
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Execution time: {elapsed:.2f}s")
        print(f"Processing speed (pairs/sec): {fps:.2f}")
        print(f"Memory: {psutil.Process().memory_info().rss / (1024**2):.2f} MB")
        if self.density_lane1:
            print(f"\nLane 1: Avg density: {np.mean(self.density_lane1):.4f}, "
                  f"Avg vehicles: {np.mean(self.vehicle_counts_lane1):.2f}")
        if self.density_lane2:
            print(f"Lane 2: Avg density: {np.mean(self.density_lane2):.4f}, "
                  f"Avg vehicles: {np.mean(self.vehicle_counts_lane2):.2f}")
        print(f"\nOutput: {output_path}")
        print("="*70)


def main():
    processor = MultiprocessingTPL()
    processor.run()


if __name__ == "__main__":
    main()
