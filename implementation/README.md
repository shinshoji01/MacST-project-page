We'll proceed with this implementation in a notebook form. We also share our docker environment for everybody to run the code.

# Notebooks
- [01_Transliteration_via_LLM.ipynb](https://github.com/shinshoji01/MacST-project-page/blob/main/implementation/notebooks/01_Transliteration_via_LLM.ipynb)
  - Transliteration via Large Language Models (LLMs)
- [02_TTS_via_Elevenlabs.ipynb](https://github.com/shinshoji01/MacST-project-page/blob/main/implementation/notebooks/02_TTS_via_Elevenlabs.ipynb)
  - Multilingual Text-to-Speech (TTS) via Elevenlabs

# Docker
I created my environment with docker-compose, so if you want to run my notebooks, please execute the following codes.
```bash
cd MacST-project-page/implementation/Docker
docker-compose up -d --build
docker-compose exec macst bash
nohup jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' --port 1234 &
```
Then, go to http://localhost:1234/lab

If you are unfamiliar with Vim, please run the following code before opening JupyterLab.
```bash
pip uninstall jupyter-vim
```

# Others
- I use my repository (`sho_util`) in the implementation. Please run the following after you `git clone`.
```
git submodule update --init --recursive
```
