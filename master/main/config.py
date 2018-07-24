import argparse


class ArgParser(object):
    def __init__(self, args):
        self.args = args

    @classmethod
    def from_cmd(cls):
        """
        This function parses and return arguments passed in
        N_particles
        m_discretization
        dimension
        N_run
        n_picard
        Number of trees
        Max Leaf nodes
        Min_samples_leaf

        :return:
        """
        # Assign description to the help doc
        parser = argparse.ArgumentParser(description='define BSDE statics from args')
        parser.add_argument(
            '-fwd', '--fwd_particles', type=int, help='number of particles used for pricing', default=1e4)
        parser.add_argument(
            '-tds', '--time_delta_steps', type=int, help='time discretization', default=2)
        parser.add_argument(
            '-dim', '--dimension', type=int, help='number of options/portfolio dimension', default=1)
        parser.add_argument(
            '-nr', '--number_run', type=int, help='Number of simulations to draw statistics', default=20)
        parser.add_argument(
            '-np', '--n_picard', type=int, help='Picard number used for Variance reduction', default=1)
        parser.add_argument(
            '-nt', '--n_trees', type=int, help='Number of trees used for Random Forest Regression ', default=100)
        parser.add_argument(
            '-mln', '--max_leaf_nodes', type=int, help='max number of leaf generated in a node for every tree',
            default=10)

        args = parser.parse_args()
        return cls(args)


if __name__ == '__main__':
    args = ArgParser.from_cmd()
    print (args.__dict__['args'])