import subprocess

def run_script(script_name):
    process = subprocess.Popen(["python", script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())
    stderr = process.communicate()[1]
    if stderr:
        print(f"Errors in {script_name}:\n{stderr}")

def main():
    scripts = [
        "contour.py",
        "ml_model.py",
        "make_3d.py"
    ]
    
    for script in scripts:
        print(f"Running {script}...")
        run_script(script)

if __name__ == "__main__":
    main()
