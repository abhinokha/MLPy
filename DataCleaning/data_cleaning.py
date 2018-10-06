## importing libraries
import numpy as np
import pandas as pd

import sys

from sklearn.preprocessing import LabelEncoder


## class for data
class Data:

    ## initialization function
    def __init__(self, train, test):
        self.train_raw = train
        self.test_raw = test
        self.train = pd.DataFrame.copy(train)
        self.test = pd.DataFrame.copy(test)


    ## function for extracting id column
    def extract_ids(self):
        """Extracts id column if present, else generates default.
        """

        # checking if first column is id
        first_column = self.train.columns[0].lower()

        id_flag = 0

        if "id" in first_column or "no" in first_column or "number" in first_column:
            while True:
                id_input = str(input("Is %s the observation id/number?(y/n): " % (str(self.train.columns[0]))).lower())
                if id_input.lower() not in ["y","n"]:
                    print("Please enter y or n")
                else:
                    print("")
                    break
                    break

            if id_input == "y":
                id_flag = 1
                id_column = self.train.columns[0]

                self.train_ids = np.array(self.train[id_column])
                self.train.drop(id_column, axis=1, inplace=True)
                print("Column %s extracted as id from train data" % (id_column))
        
                try:
                    self.test_ids = np.array(self.test[id_column])
                    self.test.drop(id_column, axis=1, inplace=True)
                    print("Column %s extracted as id from test data" % (id_column))
                except:
                    self.test_ids = np.arange(len(self.test)) + 1
                    print("Column %s not found in test data, created default ids" % (id_column))
                    
        # asking for id column
        if id_flag == 0:
            while True:
                id_column = str(input("Please enter column name of id (or type none if no such column exists): "))
                if id_column.lower() != "none" and id_column not in self.train.columns.values:
                    print("Column %s not found in train data" % (id_column))
                else:
                    print("")
                    break
                    break

            if id_column != "none":
                self.train_ids = np.array(self.train[id_column])
                self.train.drop(id_column, axis=1, inplace=True)
                print("Column %s extracted as id from train data" % (id_column))
                try:
                    self.test_ids = np.array(self.test[id_column])
                    self.test.drop(id_column, axis=1, inplace=True)
                    print("Column %s extracted as id from test data" % (id_column))
                except:
                    self.test_ids = np.arange(len(self.test)) + 1
                    print("Column %s not found in test data, created default ids" %(id_column))
            else:
                self.train_ids = np.arange(len(self.train)) + 1
                self.test_ids = np.arange(len(self.test)) + 1
            print("Created default ids for train data")
            print("Created default ids for test data")

        print("")


    ## function for extracting target variable
    def extract_target(self):
        """Extracts target variable.
        """

        target_flag = 0

        # checking if target variable is present in train data
        for colname in self.train.columns.values:
            if colname.lower() in ["response","result","target"]:
                while True:
                    target_input = str(input("Is %s the target variable?(y/n): " % (colname)))
                    if target_input not in ["y","n"]:
                        print("Please enter y or n")
                    else:
                        print("")
                        break
                        break

                if target_input == "y":
                    target_flag = 1
                    self.target = np.array(self.train[colname])
                    self.train.drop(colname, axis=1, inplace=True)
                    if colname in self.test.columns.values:
                        self.test.drop(colname, axis=1, inplace=True)
                    print("Column %s extracted as target variable from data" % (colname))

        # asking for target variable
        if target_flag == 0:
            while True:
                target_column = str(input("Please enter column name of target variable (or q to quit): "))
                if target_column == "q":
                    sys.exit()
                if target_column not in self.train.columns.values:
                    print("Column %s not found in train data" % (target_column))
                else:
                    print("")
                    break
                    break

            self.target = np.array(self.train[target_column])
            self.train.drop(target_column, axis=1, inplace=True)
            if target_column in self.test.columns.values:
                self.test.drop(target_column, axis=1, inplace=True)
            print("Column %s extracted as target variable from data" % (target_column))


    ## function for checking columns
    def check_column_names(self):
        """Checks if all columns are present and removes ones that aren"t.
        """

        train_cols = []
        test_cols = []

        # extracting columns present in train but not in test
        for colname in self.train.columns:
            if colname not in self.test.columns:
                train_cols.append(colname)

        # extracting columns present in test but not in train
        for colname in self.test.columns:
            if colname not in self.train.columns:
                test_cols.append(colname)

        # removing columns from train
        if len(train_cols) > 0:
            for i in train_cols:
                del self.train[i]
                print("Column %s not found in test data, removed from train data" % (i))

        # removing columns from test
        if len(test_cols) > 0:
            for i in test_cols:
                del self.test[i]
                print("Column %s not found in train data, removed from test data" % (i))

        self.test = self.test[self.train.columns]

        print("")


    ## function for removing constant columns
    def remove_constant_variables(self):
        """Removes all columns with constant value.
        """

        # creating panel
        panel = pd.concat([self.train, self.test], ignore_index=True)

        # removing constant columns
        for colname in self.train.columns:
            if len(np.unique(self.train[colname].values.astype("str"))) == 1:
                del panel[colname]
                print("Column %s has zero variance and is removed from data" % (colname))

        self.train, self.test = panel.loc[0:len(self.train)-1,], panel.loc[len(self.train):len(panel)-1,]

        print("")


    ## function for converting two-element columns to binary
    def convert_columns_to_binary(self):
        """Converts all columns with two elements into a binary column.
        """

        # creating panel
        panel = pd.concat([self.train, self.test], ignore_index=True)

        change = False

        # converting two-element columns to binary column
        for colname in self.train.columns:
            if len(np.unique(self.train[colname].values.astype("str"))) == 2:
                if not all(np.unique(self.train[colname].values.astype("str")) == ["0","1"]):
                    label = LabelEncoder()
                    label.fit(list(panel[colname].values.astype("str")))
                    panel[colname] = label.transform(list(panel[colname].values.astype("str")))

                    change = True
                    print("Column %s converted to binary" % (colname))

        if not change:
            print("\nNo binary columns in data")

        self.train, self.test = panel.loc[0:len(self.train)-1,], panel.loc[len(self.train):len(panel)-1,]

        print("")


    ## function for checking date variables
    def check_date_variables(self):
        """Checks for date variables
        """

        for colname in self.train.columns:

            # checking if column name has "date"
            if "date" in colname.lower():
                while True:
                    date_input = str(input("Is %s a date variable?(y/n): " % (colname)))
                    if date_input not in ["y","n"]:
                        print("Please enter y or n")
                    else:
                        break
                        break

                print("1. Extract year/month/day features\n2. Remove from data\n3. Do nothing")
                while True:
                    date_conversion = str(input("Choose any one: "))
                    if date_conversion not in ["1","2","3"]:
                        print("Please choose one of the above")
                    else:
                        print("")
                        break
                        break

                if date_conversion == "1":
                    try:
                        self.train[colname] = pd.DatetimeIndex(self.train[colname])
                        self.test[colname] = pd.DatetimeIndex(self.test[colname])

                        self.train[colname+"_Year"] = self.train[colname].dt.year
                        self.train[colname+"_Month"] = self.train[colname].dt.month
                        self.train[colname+"_Day"] = self.train[colname].dt.day
                        self.train[colname+"_Weekday"] = self.train[colname].dt.weekday
                        self.test[colname+"_Year"] = self.test[colname].dt.year
                        self.test[colname+"_Month"] = self.test[colname].dt.month
                        self.test[colname+"_Day"] = self.test[colname].dt.day
                        self.test[colname+"_Weekday"] = self.test[colname].dt.weekday
                        
                        print("Column %s converted into date features" % (colname))
                    except:
                        print("Column %s could not be converted into date features, removed from data" %(colname))

                    self.train.drop(colname, axis=1, inplace=True)
                    self.test.drop(colname, axis=1, inplace=True)
                elif date_conversion == "2":
                    self.train.drop(colname, axis=1, inplace=True)
                    self.test.drop(colname, axis=1, inplace=True)
                    print("Column %s removed from data" % (colname))

        print("")


    ## function for checking categorical variables
    def check_categorical_variables(self):
        """Checks if levels of categorical variables in train and test are consistent and removes inconsistent variables.
        """

        cols = []

        # removing columns with no common categories
        for colname in self.train.columns:
            if self.train[colname].dtype == "object":
                train_levels = np.unique(self.train[colname].values.astype("str"))
                test_levels = np.unique(self.test[colname].values.astype("str"))
                common_levels = np.intersect1d(train_levels, test_levels)

                if len(common_levels) == 0:
                    cols.append(colname)

        if len(cols) > 0:
            for i in cols:
                del self.train[i]
                del self.test[i]
                print("Column %s has no common categories in train data and test data, hence removed from data" % (i))

        print("")


    ## function for encoding categorical variables
    def encode_categories(self):
        """Encodes categorical variables into one-hot or label.
        """

        # creating panel
        panel = pd.concat([self.train, self.test], ignore_index=True)
        
        # extracting categorical variables
        categorical_variables = []

        for colname in self.train.columns:
            if self.train[colname].dtype == "object":
                categorical_variables.append(colname)
                print("Categorical Variable: %s, No. Categories: %d" % (colname, len(np.unique(self.train[colname].values.astype("str")))))

        if len(categorical_variables) > 0:
            print("1: Label encode categorical variables\n2: Onehot encode categorical variables\n3: Remove categorical variables\n4: Do nothing")
            
            while True:
                encoding = str(input("Choose any one: "))
                if encoding.lower() not in ["1", "2", "3", "4"]:
                    print("Please choose one of the above: ")
                else:
                    print("")
                    break
                    break

            if encoding == "1":
                label = LabelEncoder()
                for colname in categorical_variables:
                    label.fit(list(panel[colname].values.astype("str")))
                    panel[colname] = label.transform(list(panel[colname].values.astype("str")))
                print("Label encoded the categorical variables")
            elif encoding == "2":
                self.train = pd.get_dummies(self.train, columns=categorical_variables)
                panel = pd.get_dummies(panel, columns=categorical_variables)
                panel = panel[self.train.columns]
                print("Onehot encoded the categorical variables")
            elif encoding == "3":
                panel.drop(categorical_variables, axis=1, inplace=True)
                print("Categorical variables removed from data")

        self.train, self.test = panel.loc[0:len(self.train)-1,], panel.loc[len(self.train):len(panel)-1,]

        print("")


    ## function for cleaning data
    def clean_data(self):
        """Performs standard data cleaning functions
        """

        self.extract_ids()
        self.extract_target()

        self.check_column_names()
        
        self.remove_constant_variables()
        self.convert_columns_to_binary()

        self.check_date_variables()
        self.check_categorical_variables()
        self.encode_categories()

        print("Data is clean and ready!\n")


    ## function for removing columns
    def drop_columns(self, colnames):
        """Drops columns
        """

        self.train.drop(colnames, axis=1, inplace=True)
        self.test.drop(colnames, axis=1, inplace=True)

        for colname in colnames:
            print("Column %s removed from data" % (colname))

        print("")

