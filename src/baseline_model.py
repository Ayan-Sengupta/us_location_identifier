from load_data import load_data
from sklearn.model_selection import train_test_split

# load the data 
df = load_data('data/users_with_locations.csv.gz')

# split data for k-fold cross validation
X = df.drop(columns=['location_us'])
y = df['location_us']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shape of the data
print(f"Shape of training data: {X_train.shape}")
print(f"Shape of testing data: {X_test.shape}")
