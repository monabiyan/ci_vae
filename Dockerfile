# Dockerfile for Linux workstation querying to EDW

# Author: Mohsen Nabian

# Notes: I added in my data mart in the ini configuration lines at the end; 
# you can still query any data marts you have access to. 
# The base image is a NVIDIA tensorflow image; any ubuntu 20.04 image should work 
# if not 20.04, the curl lines may need to be updated.

# Example build line:
# docker build -f Dockerfile -t tf_query_test/test1:cashfcast1 .
# Example run line
# docker run --gpus all -it --rm -v /home/mnabian/Documents/ci_vae_singlecell:/workstation_root tf_query_test/test1:cashfcast1
# Example copy files from inside Docker to local
# docker cp romantic_meninsky:/workstation_root .
 


FROM nvcr.io/nvidia/tensorflow:20.12-tf1-py3

RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && apt-get update

#Ubuntu 20.04
RUN curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list

# MS SQL Tools
RUN apt-get update
RUN ACCEPT_EULA=Y apt-get install -y msodbcsql17
# optional: for bcp and sqlcmd
RUN ACCEPT_EULA=Y apt-get install -y mssql-tools
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bash_profile
RUN echo 'export PATH="$PATH:/opt/mssql-tools/bin"' >> ~/.bashrc
RUN source ~/.bashrc
# optional: for unixODBC development headers

# Installing odbc tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc-dev \
    unixodbc \
    libpq-dev 

# Installing freetds
RUN apt-get install -y freetds-dev freetds-bin tdsodbc

# Python SQL steps
RUN pip install sqlalchemy
RUN pip install pyodbc
RUN pip install xgboost 
RUN pip install bokeh
RUN pip	install	flask
RUN pip	install	flask-sqlalchemy
RUN pip	install	flask-restful
RUN pip	install	seaborn
RUN pip	install	treeinterpreter
RUN pip	install	dill
RUN pip	install	lightgbm
RUN pip install shap 
RUN pip install catboost 
RUN pip install --user tables
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install torch
RUN pip install torchvision
RUN pip install umap-learn
RUN pip install numba
 

# Writing out configuration information to needed ini files
RUN echo $'[FreeTDS]\nDescription=Freetds\nDriver=/usr/lib/x86_64-linux-gnu/odbc/libtdsodbc.so\nTDS Version=7.0\nFileUsage=1' >> /etc/odbcinst.ini
RUN echo $'[fi_dm_clinical_analytics]\nDriver=FreeTDS\nServer=p-wedwsqlcs\nPort=1433\nDatabase=FI_DM_CLINICAL_ANALYTICS\nTDS version=7.3\nTrace=No' >> /etc/odbc.ini
RUN echo $'[fi_dm_clinical_analytics]\nhost = p-wedwsqlcs\nport = 1433\ntds version = 7.3' >> /usr/local/etc/freetds.conf




