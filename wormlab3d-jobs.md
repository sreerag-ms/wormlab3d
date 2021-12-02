
# Wormlab3d

## 1. Calibration

**What:** Camera calibration. Find the appropriate pinhole camera model parameters for each of the 3 cameras.

**Inputs:** Calibration slide images (stored in worm\_data).

**Outputs:** Camera calibration xml files (stored in worm\_data).

**Requires: **opencv

**Files: **?

**Process:** ?



**TODO:**

   1.  Define process. What files are involved, how are xml files generated?
   1. What do we need to retain -- calibration images and output xml files?
   1. Are the xml files for a single experiment? (Which could relate to multiple runs/recordings?)
   1. Does the process require any parameters? Are these defined anywhere?


My understanding:

    - This first aspect is carried out in the worm-cv (c++). 

    - There are tests and a test data set there.

    - I would like to pull this in but I don't know how to do so structurally.





## 2. Preprocessing

### 2.1 Image normalisation

**What:** [VAIB:] For each camera we obtain a single background image per clip (typically minutes long) from the maximum values of temporary low-passed pixel intensities. The background image is subtracted from the video frames and the image intensities are then normalized to full brightness in each frame.

**Inputs:** Full resolution video clips (mp4?).

**Outputs:** Compressed video clips, background images?

**Requires:** ?

**Files:** ?

**Process: **

   1. Load a video clip. (norpix?)
   1. Obtain background image. (pushed!)
   1. For each frame; subtract background image, renormalise.
   1. Generate compressed video clip.
   1. Store background image and compressed video.


**TODO:**

   1. Check process is correct. What files/packages are involved?
   1. Specify parameters (ie, lowpass timescale). Are these defined anywhere?
   1. Are multiple background images used, ie, windowed lowpass?


### 2.2 Object tracking and triangulation

**What: **Locate and track the worm as it moves around the FOV.

**Inputs:** Compressed video clips.

**Outputs:** Zoomed-in/low-resolution video clips, offset coordinates.

**Requires:** ?

**Files: **?

**Process: **

   1. Load a compressed video clip (background-subtracted and normalised).
   1. Locate the worm centres in all frames.
   1. Smooth the centroid coordinates over time to avoid jumping.
   1. Extract fixed-size region from each frame centred at centroids.
   1. Generate zoomed-in video clip from extracted regions.
   1. Store low-res clip and relative coordinates for each frame.


**TODO:**

   1. Unclear from VAIB paper. Appears to be separate from the normalisation but no details given in the paper.
   1. Check process is correct. What files/packages are involved?
   1. Specify parameters (eg, for temporal smoothing). 




## 3. Segmentation (2D)

### 3.1 Train 2D segmentation CNN

**What:** Build and train a CNN to generate midlines from images.

**Inputs:** Dataset of images with hand-annotated midlines.

**Outputs:** Trained model checkpoint.

**Requires:** pytorch, GPU, ?

**Files:** ?

**Process:**

   1. Define CNN architecture (depth, number of filters etc).
   1. Train model (on GPU machine).
   1. Track training statistics, losses etc.
   1. Save model checkpoints.


**TODO:**

   1. Is is currently possible to parametrise the architecture and save the parameters?
   1. Do we have a high-accuracy trained model checkpoint? (If so, I propose we freeze/simplify the architecture to what we know is working well).
   1. Where are the model checkpoints stored?
   1. How large are the checkpoint files? (roughly)
   1. How is it evaluated? Is the dataset split into train/test to check for overfitting?


eval\_countour.py?



### 3.2 Generate 2D midlines.

**What:** Generate 2D midlines from video frames.

**Inputs: **Trained 2D segmentation CNN checkpoint, zoomed-in video clip.

**Outputs:** Probability of worm/not-worm for each pixel in each frame.

**Requires:** pytorch

**Files:** ?

**Process:**

   1. Load CNN checkpoint.
   1. Process each frame in clip with CNN to generate pixel-level probablistic segmentation maps.
   1. Apply a high-pass filter to each map with a cut-off on the set of candidate midline points. (??)
   1. Select the largest connected component in each frame.
   1. Save proposed 2D midlines.


**TODO:**

   1. Are there additional parameters for this process ie, for the high pass filtering? 
   1. Does this part ever fail entirely?
   1. Is the output a probability or is it binarised?
    

    

## 4. Recalibration

**What:** Adjust camera calibration parameters to maximise 3D projection overlap.

**Inputs:** Video triplets?? (Does this use the segmented images from 3.2 or video frames from 2.1/2.2?)

**Outputs:** Camera calibration xml files ?? (Or just a xy/yz shift for each frame?)

**Requires:** pytorch, GPU

**Files:** ?

**Process:**

   1. Using a sliding window of size N, construct a batch of consecutive frame-triplets.
   1. For each batch: "Using stochastic gradient descent, we optimize the three-way correlation of the brightness with respect to shifts along the local coordinate axes"
   1. Move sliding window along T frames and repeat #2.
   1. Average the overlapping results obtained at each frame.
   1. Apply temporal smoothing.
   1. Save "shifts"/updated calibrations parameters?.


**TODO:**

   1. What are the inputs/outputs here exactly? Does it need to be here in the pipeline?
   1. If this is done earlier the frames could be fixed in the preprocessing stage and shifts forgotten about.
   1. Specify parameters (N, T, smoothing, n\_iterations, etc).
   1. Does this really require a GPU to run? Or is this just to speed up throughput?




## 5. Skeletonising

**What:** Generate 3D voxels that best agree with the 2D midlines in all three views.

**Inputs:** 3x2D segmented images, shifts

**Outputs:** 3D voxels ??

**Requires:** ?

**Files:** ?

**Process:**

   1. Apply shifts to images (recalibration).
   1. In each 2D image, apply the Guo-Hall algorithm to obtain thin skeletons.
   1. Lift the skeletons into 3D, replicating along the unknown dimension.
   1. Use 3-way majority voting to determine if each voxel should be considered part of the midline.
   1. Save 3D discrete skeleton (?).


**TODO:**

   1. Unclear to me from VAIB paper exactly what the process is, the summary above needs checking.
   1. Is the output a cuboid or list of coordinates? (Ie, dense or sparse?)
   1. Are the output values binary or weighted? (The weightings are referred to in #6 but think they should be generated here).
   1. Are we in lab-coordinates here?
   1. Specify parameters.


    

## 6. Curve fitting

**What:** Generate the final 3D midline curve.

**Inputs:** Discrete 3D skeleton (voxels/voxel coordinates)

**Outputs:** Midline xyz coordinates defined at N=128 points.

**Requires:** ?

**Files:** ?

**Process:**

   1. Define the curve as a 1D mesh representing an elastic rod in a fluid with an internal stiffness using a finite element formulation. 
   1. (Control points are weighted by the product of the corresponding respective pixel intensities. )
   1. Define control points at the centres of the 3D skeleton voxels.
   1. Each control point pulls on the nearest mesh point with a spring force proportional to the distance and vice versa and scaled by the weight of the control point. 
   1. In a simulation, the model gives in to the forces, while growing to the full length. 
   1. Find the reconstructed midline as a local minimum of the energy in the stiffness and applied forces. 
   1. In subsequent frames use the shortened curves as an initial guess to propagate the curve orientation.


**TODO:**

   1. Process needs defining in a more programmatic way.
   1. Process #2 probably wants moving into #5?
   1. What is the stopping criteria? How is it simulated?
   1. Must output in lab coordinates here, so either inputs must be in lab coordinates or this process will require more inputs to work that out.
   1. Specify parameters.


    

## 7. Eigenworms

**What:** Complex PCA embedding of worm shapes.

**Inputs:** Midlines

**Outputs:** CPCA basis, midline embeddings

**Requires:** ?

**Files:** ?

**Process:**

   1. Run PCA on dataset of midline curves.
   1. Project midline curves onto the PCA basis.
   1. Save basis (and embeddings?).
    

**TODO:**

   1. Do we need to save the embeddings or just generate embeddings on demand/in preparing datasets and just save the basis?
   1. Should this output anything else? 




## 8. Trajectory datasets

**What:** Generate a dataset of fixed-length sequences of midlines.

**Inputs:** Midlines

**Outputs:** Dataset of trajectories. Optionally split into disjoint train/test sets.

**Requires:** numpy

**Files:** wormpy/nn/data\_helpers.py

**Process:**

   1. Filter the experiments as required (eg, concentration, labels, sex etc)
   1. For each matching clip, extract potentially overlapping sequences of midlines as trajectories.
   1. For train/test split, determine split point(s) in clip and throw away overlapping trajectories.
   1. Collate trajectories into train and test sets.
   1. Save the trajectory dataset(s).


**TODO:**

1. Move away from flat file datasets and reduce data duplication by using a proper database.





## 9. Clustering

**What:** Perform clustering analysis on the trajectories.

**Inputs:** Trajectory dataset.

**Outputs:** Plots, stats...?

**Requires:** scipy, numpy, matplotlib

**Files:** wormpy/gait\_classifier/cluster\_tsne.py

**Process:**

   1. Run PCA on trajectory dataset to find basis.
   1. Embed trajectories in this basis.
   1. Generate tSNE embeddings of trajectory embeddings.
   1. Generate scatter plots. 
   1. Optionally colour the points according to some labels.


**TODO:**

   1. Clustering on unlabelled data.
   1. GMM (wormpy/gait\_classifier/gmm.py)
   1. Heirarchical clustering.
    

    

## 10. Classification

### 10.1 Train classifier

**What:** Train a classifier using manually-labelled trajectories.

**Inputs:** Labelled trajectory dataset.

**Outputs:** Trained classifier checkpoint.

**Requires:** pytorch (for NN classifier), scipy, scikit-learn (other classifier models)

**Files:** wormpy/gait\_classifier/*

**Process:**

   1. Define classifier type and hyperparameters (eg. NN architecture).
   1. Train model.
   1. Track training statistics, losses etc.
   1. Save model checkpoints.


**TODO:**

   1. Improve model saving/restoring/parametrisation.


### 10.2 Generate classifications

**What:** Classify unlabelled data using trained classifier.

**Inputs:** Trajectory dataset, trained classifier checkpoint.

**Outputs:** Classification probabilities for each trajectory.

**Requires:** pytorch (for NN classifier), scipy, scikit-learn (other classifier models)

**Files:** wormpy/gait\_classifier/*

**Process:**

   1. Load classifier checkpoint.
   1. Process all trajectories in dataset to generate classifications.
   1. Save classifications.


**TODO:**

   1. Save classification outputs against trajectories.
   1. Link classifications to the constituent frames




## 11. Control inference (inverse modelling)

### 11.1 Generate controls from trajectories

**What:** Build a database of mappings between sequences of muscle activations and trajectories.

**Inputs:** Trajectory dataset

**Outputs:** Controls for each trajectory.

**Requires:** simple-worm, fenics

**Files:** simple-worm/*

**Process:**

   1. Using the simple-worm model, solve the inverse problem for each trajectory to find the optimal controls.
   1. Save optimal controls.




11.1 Train inverse model

What: Train an inverse model to produce muscle activations (controls) for a given trajectory.

Inputs: Trajectory dataset



    

    

    

    

    


