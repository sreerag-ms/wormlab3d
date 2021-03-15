configfile: "config.yaml"

rule all:
    input:
        config["cam1"]["background"],
        config["cam2"]["background"],
        config["cam3"]["background"],
        config["cam1"]["video"],
        config["cam2"]["video"],
        config["cam3"]["video"],
        config["calibration"]["output"]

rule calibration:
    """
    Camera calibration. Find the appropriate pinhole camera model parameters for each of the 3 cameras.
    Inputs: Calibration slide images (stored in worm_data)
    Outputs: Camera calibration xml files (stored in worm_data).

    TODO: how to call this calibration executable from wormcv?
    TODO: where to put tests data and test parameters
    """
    input:
        calibration_grid_directory=config["calibration"]["directory"],
        calibration_parameter_file=config["calibration"]["parameters"]
    output:
        output_file=config["calibration"]["output"]
    shell:
        """
        calibration -f \
          -p {input.calibration_grid_directory} \
          -i {input.calibration_parameter_file} \
          -o {output.output_file}
        """

rule generate_background:
    """
    Generates a fixed background image of the input video using a low pass filter.

    TODO This should be independent of the cam-id
    """
    input:
        raw_1=config["cam1"]["raw"],
        raw_2=config["cam2"]["raw"],
        raw_3=config["cam3"]["raw"],
    output:
        background_1=config["cam1"]["background"],
        background_2=config["cam2"]["background"],
        background_3=config["cam3"]["background"],
    shell:
        """
        ./wormlab3d/preprocessing/create_bg_lp.py {input.raw_1} {output.background_1}
        ./wormlab3d/preprocessing/create_bg_lp.py {input.raw_2} {output.background_2}
        ./wormlab3d/preprocessing/create_bg_lp.py {input.raw_3} {output.background_3}
        """

rule compress_video:
    """
    Regenerate a video with a static background and use lossless compression.
    TODO This should be independent of the cam-id
    TODO This file needs a proper tidy up
    """
    input:
        raw_1=config["cam1"]["raw"],
        background_1=config["cam1"]["background"],
        raw_2=config["cam2"]["raw"],
        background_2=config["cam2"]["background"],
        raw_3=config["cam3"]["raw"],
        background_3=config["cam3"]["background"],
    output:
        video_1=config["cam1"]["video"],
        video_2=config["cam2"]["video"],
        video_3=config["cam3"]["video"],
    shell:
        """
        ./wormlab3d/preprocessing/cont-movie.py \
            --if={input.raw_1} \
            --bg={input.background_1} \
            --of={output.video_1}
        ./wormlab3d/preprocessing/cont-movie.py \
            --if={input.raw_2} \
            --bg={input.background_2} \
            --of={output.video_2}
        ./wormlab3d/preprocessing/cont-movie.py \
            --if={input.raw_3} \
            --bg={input.background_3} \
            --of={output.video_3}
        """
