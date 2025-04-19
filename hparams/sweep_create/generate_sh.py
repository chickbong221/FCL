import os

# Đường dẫn thư mục chứa các config .json
sweep_dir = "../sweep_STGM_grid_split"
script_output_dir = "../sweep_STGM_scripts"
os.makedirs(script_output_dir, exist_ok=True)

for computer in sorted(os.listdir(sweep_dir)):
    computer_path = os.path.join(sweep_dir, computer)
    if not os.path.isdir(computer_path):
        continue

    for part in sorted(os.listdir(computer_path)):
        part_path = os.path.join(computer_path, part)
        if not os.path.isdir(part_path):
            continue

        script_lines = [
            "#!/bin/bash",
            f"# Sweep for: {computer}/{part}",
            "",
        ]

        for json_file in sorted(os.listdir(part_path)):
            if json_file.endswith(".json"):
                # Chuyển đường dẫn thành relative từ thư mục chạy script
                json_path = os.path.join("hparams", "sweep_STGM_grid_split", computer, part, json_file)
                command = f"python system/main.py --cfp ./{json_path} --wandb True"
                script_lines.append(command)

        # Tạo tên file script: computer1_part0.sh, computer2_part3.sh, ...
        script_name = f"{computer}_{part}.sh"
        script_file_path = os.path.join(script_output_dir, script_name)

        with open(script_file_path, "w") as f:
            f.write("\n".join(script_lines))
