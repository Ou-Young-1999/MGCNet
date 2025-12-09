import os
import numpy as np
import json
from collections import Counter


def analyze_npz_labels(folder_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    # 存储结果
    results = {}

    count_sum = Counter()

    # 遍历文件夹中的所有文件
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.npz'):
            file_path = os.path.join(folder_path, filename)
            try:
                # 加载 .npz 文件
                data = np.load(file_path, allow_pickle=True)
                print(f'{filename}:{data['segments'].shape}')

                # 假设 labels 是其中的一个键
                if 'labels' in data:
                    labels = data['labels']

                    # 确保 labels 是一维数组或列表
                    labels = np.array(labels).flatten()

                    # 统计 N 和 AFIB 的数量
                    count_N = int(np.sum(labels == 'N'))
                    count_AFIB = int(np.sum(labels == 'AFIB'))

                    # 记录结果
                    results[os.path.splitext(filename)[0]] = {
                        'N': count_N,
                        'AFIB': count_AFIB
                    }
                    count_sum.update({
                        'N': count_N,
                        'AFIB': count_AFIB
                    })
                else:
                    print(f"警告: {filename} 中没有 'labels' 键")
                    results[filename] = {'error': 'no labels key'}

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                results[filename] = {'error': str(e)}

    # 保存结果到 JSON 文件
    print(f'总计：{count_sum}')
    output_file = os.path.join(output_folder, 'label_counts.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"统计完成，结果已保存至: {output_file}")


if __name__ == "__main__":
    folder_path = './afdb_dataset/data'
    output_folder = './data_split/bihaf'

    analyze_npz_labels(folder_path, output_folder)