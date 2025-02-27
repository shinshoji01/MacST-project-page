FROM ubuntu:22.04

COPY . /build_context
WORKDIR /opt

ENV TZ Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
	    python3 \
	    python3-pip \
	    sudo \
	    wget \
	    vim \
	    espeak \
	    git

# python packages
RUN pip3 install --upgrade pip && \
	    pip3 install -r /build_context/requirements.txt

ENV HOME /opt
# activate TAB completion
RUN ipython profile create default && \
	    echo "c.Completer.use_jedi = False" >> /opt/.ipython/profile_default/ipython_config.py


# change the theme of jupyter lab
RUN mkdir -p /opt/.jupyter/lab/user-settings/@jupyterlab/apputils-extension && \
	    echo '{"theme":"JupyterLab Dark"}' > /opt/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings && \
	    mkdir -p /opt/.jupyter/lab/user-settings/@jupyterlab/notebook-extension && \
	    echo '{"codeCellConfig":{"lineNumbers":true }}' >   /opt/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings

WORKDIR /work

CMD ["bash"]
