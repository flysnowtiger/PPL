import os

dir_m1 = "./outputs/Rank_Results_Base"
dir_m2 = "./outputs/Rank_Results_PSL"
dir_m3 = "./outputs/Rank_Results_PTL"
dir_m4 = "./outputs/Rank_Results_STR"


lst = os.listdir(dir_m1)

for i in range(len(lst)):
    path_m1 = os.path.join(dir_m1, lst[i])
    path_m2 = os.path.join(dir_m2, lst[i])
    path_m3 = os.path.join(dir_m3, lst[i])
    path_m4 = os.path.join(dir_m4, lst[i])
    seq_m1 = sorted(os.listdir(path_m1))
    seq_m2 = sorted(os.listdir(path_m2))
    seq_m3 = sorted(os.listdir(path_m3))
    seq_m4 = sorted(os.listdir(path_m4))

    n = 0
    label_m1_s1 = seq_m1[n].split('_')[-1]
    label_m1_s2 = seq_m1[n + 1].split('_')[-1]
    label_m1_s3 = seq_m1[n + 2].split('_')[-1]
    # label_m1_s4 = seq_m1[n + 3].split('_')[-1]
    if label_m1_s1 == 'FALSE' and label_m1_s2 == 'FALSE':
        label_m2_s1 = seq_m2[n].split('_')[-1]
        label_m2_s2 = seq_m2[n + 1].split('_')[-1]
        label_m2_s3 = seq_m2[n + 2].split('_')[-1]
        # label_m2_s4 = seq_m2[n + 3].split('_')[-1]
        if (label_m2_s1 == 'FALSE' or label_m2_s2 == 'TRUE') and label_m2_s3 == 'TRUE':  # or :
            label_m3_s1 = seq_m3[n].split('_')[-1]
            label_m3_s2 = seq_m3[n + 1].split('_')[-1]
            label_m3_s3 = seq_m3[n + 2].split('_')[-1]
            # label_m3_s4 = seq_m3[n + 3].split('_')[-1]
            if label_m3_s1 == 'TRUE':# and label_m3_s2 == 'FALSE' and label_m2_s3 == 'TRUE':
                label_m4_s1 = seq_m4[n].split('_')[-1]
                label_m4_s2 = seq_m4[n + 1].split('_')[-1]
                label_m4_s3 = seq_m4[n + 2].split('_')[-1]
                # label_m4_s4 = seq_m4[n + 3].split('_')[-1]
                if label_m4_s1 == 'TRUE' and label_m4_s2 == 'TRUE':  # or label_m4_s4 != 'FALSE':
                    print(lst[i])
                else:
                    continue
            else:
                continue

        else:
            continue
    else:
        continue







