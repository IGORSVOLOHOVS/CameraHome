import os
import sys
import subprocess

def check_and_pull():
    """
    Runs git pull to check for updates.
    Returns (True, message) if updates were pulled, (False, message) otherwise.
    """
    try:
        # Run git fetch first
        subprocess.run(["git", "fetch"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Run git status to see if we are behind
        status_res = subprocess.run(["git", "status", "-uno"], check=True, stdout=subprocess.PIPE, text=True)
        if "Your branch is behind" in status_res.stdout or "can be fast-forwarded" in status_res.stdout:
            pull_res = subprocess.run(["git", "pull"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return True, f"Updates pulled successfully:\n{pull_res.stdout}"
        else:
            return False, "Already up to date."
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode() if e.stderr else str(e)
        return False, f"Git error: {stderr_msg}"
    except Exception as e:
        return False, f"Unexpected error checking for updates: {e}"

def restart_script():
    """
    Restarts the current Python process.
    """
    print("[INFO] Restarting script...")
    os.execv(sys.executable, [sys.executable] + sys.argv)
