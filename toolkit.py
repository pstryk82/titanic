def label_encode_vector(vector):
    label_encoder = LabelEncoder()
    vector = label_encoder.fit_transform(vector)
    return vector

def one_hot_encode_vector(vector):
    vector = vector.reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(categorical_features=[0])
    one_hot_encoded = one_hot_encoder.fit_transform(vector).toarray()
    return one_hot_encoded




#
# categorical_indices = [0, 2, 6, 7]
# for index in categorical_indices:
#     X[:, index] = toolkit.label_encode_vector(X[:, index])
#
#
#
# one_hot_encoded_replacements = {}
# for index in categorical_indices:
#     one_hot_encoded_replacements[index] = toolkit.one_hot_encode_vector(X[:, index])
#
#
# for index in reversed(categorical_indices):
#     np.delete(X, index, axis=1)
#     np.insert(X, one_hot_encoded_replacements[index], axis=1)
