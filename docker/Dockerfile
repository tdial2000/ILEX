# Use Ubuntu as a parent image
FROM python:3.9

LABEL maintainer="Tyson Dial tdial@swin.edu.au"

# install vim text editor
RUN apt update && apt install -y vim

# install python packages frm requirements!
COPY ILEX/docker/requirements.txt /tmp/
RUN cd /tmp && pip install --no-cache-dir -r requirements.txt
RUN apt update && apt-get install -y python3-tk
RUN apt update && apt install -y xauth
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : tkAgg" > /root/.config/matplotlib/matplotlibrc

# install ILEX
RUN mkdir /src && mkdir /src/ILEX
COPY ILEX /src/ILEX
RUN cd /src/ILEX && bash install_ilex.sh

# Set up .bashrc file for root user [default]
COPY ILEX/docker/rootbashrc /tmp/
RUN touch /root/.bashrc && cat /tmp/rootbashrc >> /root/.bashrc

CMD ["bash"]



