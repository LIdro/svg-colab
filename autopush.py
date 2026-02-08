import os
import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# This script watches the colab_demo directory and automatically pushes changes to GitHub.
REPO_DIR = os.path.abspath(os.path.dirname(__file__))
BRANCH = "main"
DEBOUNCE_SECONDS = 5.0  # Increased slightly for safety
LAST_EVENT = 0.0

def run(cmd):
    # Pass input=None to prevent blocking on interactive prompts
    return subprocess.run(cmd, cwd=REPO_DIR, check=False, capture_output=True, text=True)

def git_dirty():
    r = run(["git", "status", "--porcelain"])
    return r.returncode == 0 and r.stdout.strip() != ""

def commit_and_push():
    print(f"[{time.strftime('%H:%M:%S')}] Change detected. Preparing to sync...")
    
    # Check if we can push (basic check)
    run(["git", "add", "-A"])
    if not git_dirty():
        print("Nothing to commit.")
        return

    # Try to pull first to avoid conflicts
    print("Pulling latest...")
    run(["git", "pull", "--rebase", "origin", BRANCH])

    msg = f"WIP auto-sync {time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(f"Committing: {msg}")
    run(["git", "commit", "-m", msg])
    
    print("Pushing to origin...")
    r = run(["git", "push", "origin", BRANCH])

    if r.returncode != 0:
        print("Push failed. You might need to authenticate or fix a conflict manually.")
        print("Error:", r.stderr)
    else:
        print("Successfully pushed changes.")

class Handler(FileSystemEventHandler):
    def on_any_event(self, event):
        global LAST_EVENT
        # Ignore .git folder and the script itself
        path = event.src_path.replace("\\", "/")
        if "/.git/" in path or "autopush.py" in path:
            return
        if event.src_path.endswith((".pyc", ".tmp", ".log")):
            return
        
        # Only trigger on file creation, modification or deletion
        if not event.is_directory:
            LAST_EVENT = time.time()

def main():
    global LAST_EVENT
    print(f"Starting autopush watcher for: {REPO_DIR}")
    print(f"Target Branch: {BRANCH}")
    
    # Ensure watchdog is installed or warn
    try:
        import watchdog
    except ImportError:
        print("Error: 'watchdog' package not found. Install it with: pip install watchdog")
        return

    event_handler = Handler()
    observer = Observer()
    observer.schedule(event_handler, REPO_DIR, recursive=True)
    observer.start()
    print(f"Watching... (debounce={DEBOUNCE_SECONDS}s). Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1.0)
            if LAST_EVENT and (time.time() - LAST_EVENT) > DEBOUNCE_SECONDS:
                if git_dirty():
                    commit_and_push()
                LAST_EVENT = 0.0
    except KeyboardInterrupt:
        observer.stop()
        print("\nWatcher stopped.")
    observer.join()

if __name__ == "__main__":
    main()
