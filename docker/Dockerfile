# Use Ubuntu as a parent image
FROM python:3.9

LABEL maintainer="Tyson Dial tdial@swin.edu.au"

# install vim text editor
RUN apt update && apt install -y vim

# install python packages frm requirements!
COPY docker/requirements.txt /tmp/
RUN cd /tmp && pip install --no-cache-dir -r requirements.txt

# install ILEX
RUN mkdir /src && mkdir /src/ILEX
COPY ../ILEX /src/ILEX
RUN cd /src/ILEX && bash install_ilex.sh

# Set up .bashrc file for root user [default]
COPY docker/rootbashrc /tmp/
RUN touch /root/.bashrc && cat /tmp/rootbashrc >> /root/.bashrc

CMD ["bash"]



