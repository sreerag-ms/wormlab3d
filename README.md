# Wormlab3D


# Installation

First, clone the repository to your local machine and `cd` into it:

```bash
git clone git@gitlab.com:tom0/wormlab3d.git
cd wormlab3d
````

Then you will need a Python 3.9 environment. You can create one with [Conda](https://conda.io/docs/) using the environment file provided:

```bash
conda env create -f environment.yml
```

Alternatively you can install into an existing conda or pip environment using pip to handle the dependencies:

```bash
pip install -e . 
```

## Configuration (`.env`)
This project uses environment variables for configuration and settings. The `dotenv` python package is used to load settings from a `.env` file in the root of the repository and treat these as environment variables. An example [`.env.sample`](.env.sample) is included which contains the required settings; the database connection parameters.

Copy `.env.sample` to `.env`. Change the username and passwords to something more secure and then ensure the file permissions on `.env` are suitably restrictive (`600` should do).

This `.env` file will now contain all of your configuration settings. Most paths and settings can be overridden by adding lines to this file. For a full list of all the available settings and some description on each one, look in [\_\_init\_\_.py](./wormlab3d/__init__.py). Any line which contains `os.getenv(..)` describes a setting you can change by adding it to your `.env` file, for example: `LOG_LEVEL=DEBUG`.



## Creating a database

The project uses a [MongoDB](https://www.mongodb.com/) database server for all stages of the pipeline. To ease set up of a new database a `docker-compose.yml` file is provided, although you may set up the database any other way. If so, skip ahead to the configuration.

Otherwise, you will need to install both Docker Engine ([https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)) and Docker Compose ([https://docs.docker.com/compose/install/](https://docs.docker.com/compose/install/)). 

Once these are installed, from the project root run:

```bash
docker compose up [-d]
```

The `-d` flag is optional and will run the service in `daemon` mode (ie, the background). Database data is stored in `[PROJECT_ROOT]/data/db`. When you are done, you can shut the services down with:

```bash
docker compose down
```

(The database data will persist!)

### Mongo Express
The `docker-compose.yml` also starts an instance of [Mongo Express](https://github.com/mongo-express/mongo-express). This is a simple web-based admin interface for the database and can be accessed at [http://localhost:8081](http://localhost:8081).


## SSH tunnels
To connect to an external database which is not exposed to the internet, but to which you have SSH access you can use an SSH tunnel.

```bash
ssh -L 27018:localhost:27017 remote-server.net
```

This command opens an SSH connection to `remote-server.net` and forwards any connection to port 27018 on your local machine to port 27017 on `remote-server.net`.

This works for Mongo Express too, you just need to change the port;

```bash
ssh -L 8082:localhost:8081 remote-server.net
```

If Mongo Express is running on `remote-server.net:8081` you will now be able to access it at [http://localhost:8082](http://localhost:8082).
