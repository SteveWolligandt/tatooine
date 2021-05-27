#-------------------------------------------------------------------------------
class VecPrinter:
#-------------------------------------------------------------------------------
    def __init__(self, val, size):
        self.val = val
        self.size = size
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def to_string(self):
        out = '{' + str(self.val['m_data']['_M_elems'][0])
        for i in range(1, self.size):
            out = out + ', ' + str(self.val['m_data']['_M_elems'][i])
        out = out + '}'
        return out

#-------------------------------------------------------------------------------
class RayPrinter:
#-------------------------------------------------------------------------------
    def __init__(self, val, size):
        self.val = val
        self.size = size
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def to_string(self):
        out = 'origin = {' + str(self.val['m_origin']['m_data']['_M_elems'][0])
        for i in range(1, self.size):
            out = out + ', ' + str(self.val['m_origin']['m_data']['_M_elems'][i])
        out = out + '}, '
        out = out + 'direction = {' + str(self.val['m_direction']['m_data']['_M_elems'][0])
        for i in range(1, self.size):
            out = out +  ', ' + str(self.val['m_direction']['m_data']['_M_elems'][i])
        out = out + '}'
        return out
#===============================================================================
def tat_pp(val):
    if str(val.type) == 'tatooine::vec<double, 2>':    return VecPrinter(val, 2)
    if str(val.type) == 'tatooine::vec<float, 2>':     return VecPrinter(val, 2)
    if str(val.type) == 'tatooine::tensor<double, 2>': return VecPrinter(val, 2)
    if str(val.type) == 'tatooine::tensor<float, 2>':  return VecPrinter(val, 2)
    if str(val.type) == 'tatooine::vec<double, 3>':    return VecPrinter(val, 3)
    if str(val.type) == 'tatooine::vec<float, 3>':     return VecPrinter(val, 3)
    if str(val.type) == 'tatooine::tensor<double, 3>': return VecPrinter(val, 3)
    if str(val.type) == 'tatooine::tensor<float, 3>':  return VecPrinter(val, 3)
    if str(val.type) == 'tatooine::vec<double, 4>':    return VecPrinter(val, 4)
    if str(val.type) == 'tatooine::vec<float, 4>':     return VecPrinter(val, 4)
    if str(val.type) == 'tatooine::tensor<double, 4>': return VecPrinter(val, 4)
    if str(val.type) == 'tatooine::tensor<float, 4>':  return VecPrinter(val, 4)

    if str(val.type) == 'tatooine::ray<double, 3>':    return RayPrinter(val, 3)
    if str(val.type) == 'tatooine::ray<float, 3>':     return RayPrinter(val, 3)

gdb.pretty_printers.append(tat_pp)
