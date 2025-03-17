import os
import subprocess
import argparse
import time
import sys

def run_command(command, description):
    """
    Run a shell command with a description
    """
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Add current directory to PYTHONPATH for module imports
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = os.getcwd() + os.pathsep + env['PYTHONPATH']
    else:
        env['PYTHONPATH'] = os.getcwd()
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        env=env
    )
    
    # Print output in real time
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    end_time = time.time()
    
    if process.returncode != 0:
        print(f"Error executing: {command}")
        return False
    
    print(f"\nCompleted in {(end_time - start_time):.2f} seconds")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run transformer price calculator pipeline')
    parser.add_argument('--skip-data', action='store_true', help='Skip data generation')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--run-app', action='store_true', help='Run the web application after training')
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Step 1: Generate synthetic data (if not skipped)
    if not args.skip_data:
        if not run_command("python src/data_generation.py", "Generating synthetic data"):
            return
    
    # Step 2: Train models (if not skipped)
    if not args.skip_training:
        if not run_command("python src/model_training.py", "Training and evaluating models"):
            return
    
    # Step 3: Show prediction example
    run_command("python src/predict.py", "Example prediction")
    
    # Step 4: Run the web application (if specified)
    if args.run_app:
        run_command("streamlit run app.py", "Running web application")
        
    print("\nPipeline completed successfully!")
    print("To run the web application: streamlit run app.py")

if __name__ == "__main__":
    main() 