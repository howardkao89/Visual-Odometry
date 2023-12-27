import open3d as o3d
import numpy as np
import cv2 as cv
import os, argparse, glob
import multiprocessing as mp


class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params["K"]
        self.dist = camera_params["dist"]

        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, "*.png"))))

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue,))
        p.start()

        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    # Insert new camera pose here using vis.add_geometry()
                    lineset_points = []
                    points = [[319, 179, 1], [0, 0, 1], [639, 0, 1], [639, 359, 1], [0, 359, 1],]

                    for idx, vertex in enumerate(points):
                        vertex = np.linalg.pinv(self.K) @ np.reshape(vertex, (3, -1))
                        vertex = R @ vertex.reshape(3, 1) + t.reshape(3, 1)
                        vertex = np.squeeze(vertex)
                        if idx == 0:
                            lineset_points.append(2 * vertex - np.squeeze(t))
                        else:
                            lineset_points.append(vertex)

                    lineset_lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1],]
                    lineset_colors = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],]

                    lineset = o3d.geometry.LineSet()
                    lineset.points = o3d.utility.Vector3dVector(np.array(lineset_points).reshape(-1, 3))
                    lineset.lines = o3d.utility.Vector2iVector(np.array(lineset_lines).reshape(-1, 2))
                    lineset.colors = o3d.utility.Vector3dVector(np.array(lineset_colors).reshape(-1, 3))

                    vis.add_geometry(lineset)
                    pass
            except:
                pass

            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def process_frames(self, queue):
        # Initiate ORB detector
        orb = cv.ORB_create()

        # Create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        R_rel_past, t_rel_past = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)

        img_past = cv.imread(self.frame_paths[0])
        img_past = cv.undistort(img_past, self.K, self.dist)
        # Find the keypoints and descriptors with ORB
        kp_past, des_past = orb.detectAndCompute(img_past, None)

        for frame_path in self.frame_paths[1:]:
            # Compute camera pose here
            img_now = cv.imread(frame_path)
            img_now = cv.undistort(img_now, self.K, self.dist)
            # Find the keypoints and descriptors with ORB
            kp_now, des_now = orb.detectAndCompute(img_now, None)

            # Match descriptors.
            matches = bf.match(des_past, des_now)

            distance = np.array([m.distance for m in matches])
            Q1 = np.quantile(distance, 0.25)
            Q3 = np.quantile(distance, 0.75)
            IQR = Q3 - Q1
            good_matches = []
            for m in matches:
                if m.distance >= Q1 - 1.5 * IQR and m.distance <= Q3 + 1.5 * IQR:
                    good_matches.append(m)

            # Sort them in the order of their distance.
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            points_past = np.array([kp_past[m.queryIdx].pt for m in good_matches])
            points_now = np.array([kp_now[m.trainIdx].pt for m in good_matches])

            E, mask = cv.findEssentialMat(points_past, points_now, self.K)
            _, R, t, _, triangulatedPoints = cv.recoverPose(E, points_past, points_now, self.K, distanceThresh=1000, mask=mask)
            triPoints_now = []
            for x, y, z, w in triangulatedPoints.reshape(-1, 4):
                triPoints_now.append([x / w, y / w, z / w])

            if frame_path == self.frame_paths[1]:
                scale_now = np.linalg.norm(t)
            else:
                points_now_past = points_past
                triPoint_matches = []
                for i in range(len(points_past_now)):
                    for j in range(len(points_now_past)):
                        if np.all(points_past_now[i] == points_now_past[j]):
                            triPoint_matches.append([triPoints_past[i], triPoints_now[j]])

                probable_scales = []
                for i in range(len(triPoint_matches)):
                    for j in range(len(triPoint_matches)):
                        if i == j or triPoint_matches[i][1] == triPoint_matches[j][1] or triPoint_matches[i][0] == triPoint_matches[j][0]:
                            continue
                        norm_past = np.linalg.norm(np.subtract(triPoint_matches[i][0], triPoint_matches[j][0]))
                        norm_now = np.linalg.norm(np.subtract(triPoint_matches[i][1], triPoint_matches[j][1]))
                        probable_scales.append(scale_past * norm_now / norm_past)

                scale_now = np.sqrt(np.median(probable_scales))
                t *= scale_now

            R_rel_now = R_rel_past @ R
            t_rel_now = t_rel_past + R_rel_past @ t

            queue.put((R_rel_now, t_rel_now))

            points_now_draw = [kp_now[m.trainIdx] for m in good_matches]
            img_now_matches = cv.drawKeypoints(img_now, points_now_draw, None, color=(0, 255, 0))
            cv.imshow("frame", img_now_matches)
            if cv.waitKey(30) == 27:
                break

            kp_past, des_past = kp_now, des_now
            points_past_now = points_now
            triPoints_past = triPoints_now
            scale_past = scale_now
            R_rel_past, t_rel_past = R_rel_now, t_rel_now


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="directory of sequential frames")
    parser.add_argument(
        "--camera_parameters",
        default="camera_parameters.npy",
        help="npy file of camera parameters",
    )
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
