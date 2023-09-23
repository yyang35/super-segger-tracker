import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat, loadmat


class SeggerTracker:

    DIGIT_CONST = 100000

    @staticmethod
    def compress_xy(x, y):
        return y * SeggerTracker.DIGIT_CONST + x

    @staticmethod
    def extract_xy(compress):
        return [compress % SeggerTracker.DIGIT_CONST, int(compress / SeggerTracker.DIGIT_CONST)]

    @staticmethod
    def find_skeleton_centers(cell_count, gradient_total, threshold, regs_label):
        centers = []
        for i in range(cell_count + 1):
            masked = (regs_label == i) & (gradient_total < threshold)
            if not np.any(masked):
                continue
            # Skeleton of this cell
            y = np.ma.array(gradient_total * masked)
            row_indices, col_indices = np.where(y < 0)
            if len(row_indices) < 1:
                original_field = (regs_label == i)
                row_indices, col_indices = np.ma.where(original_field)

            # Find the geometry center
            x_center = np.mean(row_indices)
            y_center = np.mean(col_indices)

            # Projected the geometry center on skeleton
            distances = np.sqrt((row_indices - x_center)**2 + (col_indices - y_center)**2)
            closest_index = np.argmin(distances)
            closest_point = (row_indices[closest_index], col_indices[closest_index])

            # Found skeleton center
            centers.append(SeggerTracker.compress_xy(closest_point[1], closest_point[0]))

        return centers

    @staticmethod
    def flow_to_ends(start_points, gradient_total, threshold):
        # Don't pollute original data
        start_points_copy = start_points.copy()
        gradient_total_copy = gradient_total.copy()
        masked_array = gradient_total_copy < threshold
        # Point set is the set flow go through
        point_set = set(start_points_copy)
        end_point = []
        while point_set:
            new_set = set()
            for element in point_set:
                [xx, yy] = SeggerTracker.extract_xy(element)
                # DO waterfront
                waterfront_set = set([
                    element - 1,
                    element + 1,
                    element - SeggerTracker.DIGIT_CONST + 1,
                    element + SeggerTracker.DIGIT_CONST + 1,
                    element + SeggerTracker.DIGIT_CONST - 1,
                    element - SeggerTracker.DIGIT_CONST - 1,
                    element + SeggerTracker.DIGIT_CONST,
                    element - SeggerTracker.DIGIT_CONST,
                ])

                flag = False
                for each in waterfront_set:
                    [x, y] = SeggerTracker.extract_xy(each)
                    if (x < 0) or (y < 0) or (y >= len(masked_array)) or (x >= len(masked_array[0])):
                        continue
                    else:
                        if masked_array[y, x]:
                            new_set.add(each)
                            flag = True
                if flag is False:
                    end_point.append(element)
                masked_array[yy, xx] = False

            # Mask the pixels out when the whole waterfront of pixels iterated, so won't self eated.
            for each in new_set:
                [x, y] = SeggerTracker.extract_xy(each)
                masked_array[y, x] = False

            point_set = new_set

        return end_point

    @staticmethod
    def get_masked_skeleton(cell_number, gradient_total, regs_label, threshold):
        masked = (gradient_total < threshold)
        if cell_number != -1:
            masked = masked & (regs_label == cell_number)
        masked_skeleton = np.ma.array(gradient_total * masked)
        return masked_skeleton

    @staticmethod
    def find_multi_ends(ends_count, cell_count, centers, gradient_total, threshold, regs_label):
        all_ends = []
        for time in range(ends_count):
            current_start = centers if time == 0 else all_ends
            result_ends = SeggerTracker.flow_to_ends(current_start, gradient_total, threshold)
            for i in range(1, cell_count+1):
                mask = (gradient_total < threshold) & (regs_label == i)
                current_result = -1
                for element in result_ends:
                    [x, y] = SeggerTracker.extract_xy(element)
                    if mask[y, x]:
                        current_result = element
                if current_result != -1:
                    all_ends.append(current_result)
        return all_ends

    @staticmethod
    def calculate_local_slop(point, radius, gradient_total, regs_label, threshold):
        [x, y] = SeggerTracker.extract_xy(point)
        # +1 here is only for boundary condition
        x_array = []
        y_array = []
        cell_number = regs_label[y, x]
        mask = (gradient_total < threshold) & (regs_label == cell_number)
        for xi in range(max(x-radius, 0), min(x+radius+1, len(gradient_total[0]))):
            for yi in range(max(y-radius, 0), min(y+radius+1, len(gradient_total))):
                if (np.sqrt((xi-x)**2 + (yi-y)**2) <= radius) and (mask[yi, xi]):
                    x_array.append(xi)
                    y_array.append(yi)
        return [x - np.mean(x_array), y - np.mean(y_array)]

    @staticmethod
    def work(npy_path, threshold, slop_radius):
        dat = np.load(npy_path, allow_pickle=True).item()
        gradientsx = np.gradient(dat['flows'][-1][0])
        gradientsy = np.gradient(dat['flows'][-1][1])
        gradient_total = gradientsy[1] + gradientsx[0]
        regs_label = dat['masks']
        cell_num = np.max(regs_label)
        centers = SeggerTracker.find_skeleton_centers(cell_num, gradient_total, threshold, regs_label)
        all_ends = SeggerTracker.find_multi_ends(2, cell_num, centers, gradient_total, threshold, regs_label)
        trans_array = np.zeros([len(regs_label[0]), len(regs_label), 3])
        for each in all_ends:
            [x, y] = SeggerTracker.extract_xy(each)
            [vx, vy] = SeggerTracker.calculate_local_slop(each, slop_radius, gradient_total, regs_label, -1)
            trans_array[x, y] = [vx, vy, 1]
        data = {'vector': trans_array, 'skeleton': gradient_total}
        mat_filename = npy_path.replace('_seg.npy', '_track.mat')
        savemat(mat_filename, data)

    @staticmethod
    def save_img(mat_filename):
        loaded_data = loadmat(mat_filename)
        vector = loaded_data['vector']
        skeleton = loaded_data['skeleton']

        plt.figure(figsize=(25.63, 21.87))
        plt.imshow(skeleton, cmap='Greys')

        plt.xlim(0, 2560)
        plt.ylim(0, 2160)

        plt.gca().invert_yaxis()

        for x in range(len(vector)):
            for y in range(len(vector[x])):
                vector_field_dot = vector[x][y]
                if(vector_field_dot[2] != 0):
                    plt.scatter(x, y, c='red', marker='o', s=10)
                    vx, vy = vector_field_dot[0], vector_field_dot[1]
                    plt.arrow(x, y, vx*5, vy*5, head_width=5, width=0.5, ec='black')

        image_filename = mat_filename.replace('_track.mat', '.png')
        image_filename = image_filename.replace('xy0/phase', 'images')
        plt.savefig(image_filename)

    @staticmethod
    def generate_img(npy_path, threshold, slop_radius):
        SeggerTracker.work(npy_path, threshold, slop_radius)
        mat_filename = npy_path.replace('_seg.npy', '_track.mat')
        SeggerTracker.save_img(mat_filename)
