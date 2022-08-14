class CutStoreGen(object):

    def __init__(self):
        self.cut_storage = []
        self.const_cut_storage = []

    def store_cut(self, cut_num, node_num, local_sol, local_obj, local_grad, local_eig = 0):
        cut_info = {
            'cut_id': cut_num,
            'node_id': node_num,
            'x': local_sol,
            'fx': local_obj,
            'gx': local_grad,
            'eig': local_eig
        }
        self.cut_storage.append(cut_info)

    def store_const_cut(self, sol, const_eval, const_grad, const_eig = 0):
        cut_info = {
            'x': sol,
            'gx': const_eval,
            'ggx': const_grad,
            'eig_g': const_eig
        }

        self.const_cut_storage.append(cut_info)
