import pickle

with open('pdbbind_screening_allow_dict.pkl', 'rb') as f:
    allow_dict = pickle.load(f)


print(allow_dict)