import os

sweep_dir = "../sweep_STGM"  # nằm ở hparams
script_output_dir = "../sweep_STGM_scripts"
os.makedirs(script_output_dir, exist_ok=True)

for param_folder in os.listdir(sweep_dir):
    param_path = os.path.join(sweep_dir, param_folder)
    if not os.path.isdir(param_path):
        continue

    script_lines = [
        "#!/bin/bash",
        f"# Sweep for: {param_folder}",
        "",
    ]

    for json_file in sorted(os.listdir(param_path)):
        if json_file.endswith(".json"):
            json_path = os.path.join("hparams", "sweep_STGM", param_folder, json_file)
            command = f"python system/main.py --cfp ./{json_path} --wandb True"
            script_lines.append(command)

    # Save to .sh file
    script_file_path = os.path.join(script_output_dir, f"{param_folder}.sh")
    with open(script_file_path, "w") as f:
        f.write("\n".join(script_lines))
