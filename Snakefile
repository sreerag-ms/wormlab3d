configfile: "config.yaml"

cams = [f'cam{j}' for j in range(1, 4)]

def swap_key(value: str, old_key: str, new_key: str) -> str:
    for cam in cams:
        v = config[cam]
        if v[old_key] == value:
            new_v = v[new_key]
            return new_v

rule all:
    input:
        [config[f'{cam}']['video'] for cam in cams],
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
        # calibration_parameter_file=config["calibration"]["parameters"]
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
    """
    input:
        raw=lambda w: \
            swap_key(f'{w.background}.png', "background", "raw")
    output:
        background="{background}.png"
    shell:
        """
        ./wormlab3d/preprocessing/create_bg_lp.py {input.raw} {output.background}
        
        """

rule compress_video:
    """
    Regenerate a video with a static background and use lossless compression.
    """
    input:
        raw=lambda w: \
            swap_key(f'intermediate-data/{w.video}.avi', "video", "raw"),
        background=lambda w: \
            swap_key(f'intermediate-data/{w.video}.avi', "video", "background")
    output:
        video="intermediate-data/{video}.avi"
    shell:
        """
        ./wormlab3d/preprocessing/cont-movie.py \
            --if={input.raw} \
            --bg={input.background} \
            --of={output.video}
        """
