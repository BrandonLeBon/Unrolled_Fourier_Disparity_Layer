from Runners.Trainer import Trainer
from Runners.Tester import Tester

import yaml
import argparse

''' Read console arguments '''
def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dataset", dest="training_dataset", type=str)   
    parser.add_argument("--validation_dataset", dest="validation_dataset", type=str)
    parser.add_argument("--testing_dataset", dest="testing_dataset", type=str)
    parser.add_argument("--config", dest="config", type=str)
    parser.add_argument("--model_name", dest="model_name", type=str)
    parser.add_argument("--save_directory", dest="save_directory", type=str)
    parser.add_argument("--mode", dest="mode", type=str)
    parser.add_argument("--output_type", dest="output_type", default="views", type=str)
    args = parser.parse_args()
    return args
    
''' Open and return a yaml config file '''
def open_yaml_file(file):
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    return config
    
if __name__ == "__main__":
    args = read_arguments()
    config = open_yaml_file(args.config)
    
    if args.mode == "test":
        print("\nPREPARING TESTING")
        testing_dataset = args.testing_dataset
        tester = Tester(config, testing_dataset, args.model_name, args.save_directory, args.output_type)
        tester.forward()
    elif args.mode == "train":
        print("\nPREPARING TRAINING")
        training_dataset = args.training_dataset
        validation_dataset = args.validation_dataset
        trainer = Trainer(config, training_dataset, validation_dataset, args.model_name)
        trainer.forward()
    else:
        print("Unknown mode")
    