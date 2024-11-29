from flask import Flask, request, jsonify # type: ignore

from tensorflow.keras.models import load_model # type: ignore

import numpy as np # type: ignore
import cv2 # type: ignore
import io # type: ignore
from PIL import Image # type: ignore

import time # type: ignore

import firebase_admin # type: ignore
from firebase_admin import credentials, db # type: ignore
import os # type: ignore

from paho.mqtt.client import Client # type: ignore
import base64 # type: ignore
