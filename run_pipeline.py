"""
运行完整的金融信息抽取模型微调和评估流程
"""
import os
import sys
import time
import argparse
import subprocess

def execute_command(command, description, exit_on_error=True):
    """执行命令并打印输出"""
    print(f"\n\033[1;36m=== {description} ===\033[0m")
    print(f"执行命令: {command}")
    start_time = time.time()
    
    # 运行命令
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # 打印输出
    if result.stdout:
        print("\n标准输出:")
        print(result.stdout)
    
    if result.stderr:
        print("\n标准错误:")
        print(result.stderr)
    
    elapsed_time = time.time() - start_time
    print(f"\n耗时: {elapsed_time:.2f} 秒")
    
    if result.returncode != 0:
        print(f"\033[1;31m命令失败，退出代码: {result.returncode}\033[0m")
        if exit_on_error:
            sys.exit(result.returncode)
        return False
    else:
        print(f"\033[1;32m命令成功完成\033[0m")
        return True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="金融信息抽取工作流")
    parser.add_argument('--skip_data_prep', action='store_true', help="跳过数据准备步骤")
    parser.add_argument('--skip_training', action='store_true', help="跳过训练步骤")
    parser.add_argument('--skip_evaluation', action='store_true', help="跳过评估步骤")
    parser.add_argument('--test_file', type=str, default="sample_data5.json", help="测试文件")
    parser.add_argument('--epochs', type=int, default=5, help="训练轮数")
    parser.add_argument('--interactive', action='store_true', help="运行交互式推理")
    return parser.parse_args()

def setup_directories():
    """设置必要的目录"""
    os.makedirs("data", exist_ok=True)
    os.makedirs("trained_model", exist_ok=True)
    os.makedirs("evaluation_results", exist_ok=True)

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置目录
    setup_directories()
    
    print("\033[1;35m=== 金融信息抽取工作流 ===\033[0m")
    
    # Step 1: 数据准备
    if not args.skip_data_prep:
        execute_command("python prepare_data.py", "准备数据")
    else:
        print("跳过数据准备步骤")
    
    # Step 2: 模型微调
    if not args.skip_training:
        # 如果需要修改训练轮数，可以修改finetune_qwen_lora.py文件
        if args.epochs != 5:
            execute_command(f"sed -i 's/EPOCHS = 5/EPOCHS = {args.epochs}/g' finetune_qwen_lora.py", 
                        "修改训练轮数", exit_on_error=False)
        
        execute_command("python finetune_qwen_lora.py", "微调模型")
    else:
        print("跳过模型微调步骤")
    
    # Step 3: 模型评估
    if not args.skip_evaluation:
        if os.path.exists("trained_model"):
            execute_command(f"python evaluate.py --test_file {args.test_file}", "评估模型")
        else:
            print("警告: 未找到训练模型，跳过评估步骤")
    else:
        print("跳过模型评估步骤")
    
    # Step 4: 交互式推理（如果请求）
    if args.interactive:
        if os.path.exists("trained_model"):
            print("\n\033[1;36m=== 启动交互式推理 ===\033[0m")
            os.system("python inference.py --mode interactive")
        else:
            print("警告: 未找到训练模型，无法启动交互式推理")
    
    print("\n\033[1;35m=== 工作流完成 ===\033[0m")

if __name__ == "__main__":
    main()
