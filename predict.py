# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, File
from reconstruct_cog import reconstruct 

from typing import Any
# import unicorn here


class Output(BaseModel):
    file: File


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        s: str
    ) -> Any:
        """Run a single prediction on the model"""
        try:
            reconstruct()
            return "Hekko World Sample cog"
            # return Output(file=talknet_predict.generate_audio(voice + "|default", None, s, [], 0, None, None, None))
        except Exception as e:
            return f"Error: {e}"