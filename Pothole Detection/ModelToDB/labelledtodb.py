import firebase_admin
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import itertools
from firebase_admin import credentials
from firebase_admin import db
from reconstruct import get_locs

def save_to_db():
    data = get_locs()

    

save_to_db()