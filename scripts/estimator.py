class PoseEKF:


    def __init__(self, initial_pose, initial_covariance, geometry):


        raise NotImplementedError("Initialise the EKF and noise assumptions.")

    def predict(self, encoder_measurement):

        raise NotImplementedError("Adapt tutorial prediction to wheel encoders.")

    def correct(self, camera_observation):

        raise NotImplementedError("Adapt the tutorial measurement update.")

    def get_estimate(self):

        raise NotImplementedError("Return the current estimated state.")
