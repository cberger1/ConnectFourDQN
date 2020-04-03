import os, webbrowser


PORT = 6006 # Default TensorBoard Port


if __name__ == "__main__":
	webbrowser.open_new_tab(f"http://localhost:{PORT}/")
	os.system(f"start /wait cmd /c tensorboard --logdir=logs --port={PORT}")