import os
import sys
import click
import importlib


@click.command()
@click.argument('pattern', required = True, nargs=1, type=str)
def main(pattern):

    path = os.path.join('test_sets', pattern)
    #If there is not a set with the required pattern, we print an error
    if not os.path.isdir(path):
        message = '''
        ---------------------------------------------------------------------------------
        This pattern is not implemented. Please check the following:
            - There is a labeled test set for the required pattern.
            - There is a accuracy test subclass implemented for this pattern.
            - The pattern name is well written.
        -----------------------------------------------------------------------------
        '''
        print(message)
        sys.exit()
    else:

        # We test the accuracy for this pattern
        class_name = pattern.capitalize() + '_test'
        module = importlib.import_module('pattern_accuracy_test.' + pattern + '_test')
        pattern_class = getattr(module, class_name)

        generator_test = pattern_class(pattern)
        generator_test.pattern_detection()


########################################


if __name__ == '__main__':
    main()