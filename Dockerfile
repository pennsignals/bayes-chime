FROM continuumio/miniconda3
WORKDIR /chime_sims
RUN conda update -yq -n base -c defaults conda
RUN conda create -yq -n chime_sims python=3.7 pip pandas matplotlib scipy numpy seaborn
COPY ./requirements.txt .
RUN pip install -q -r requirements.txt
