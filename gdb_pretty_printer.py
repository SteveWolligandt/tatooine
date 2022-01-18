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

#-------------------------------------------------------------------------------
class HandlePrinter:
#-------------------------------------------------------------------------------
  def __init__(self, val):
     self.val = val
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  def to_string(self):
    return str(self.val['i'])

#===============================================================================
def tatooine_pretty_printers(val):
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

  if str(val.type) == 'tatooine::handle<tatooine::pointset<double, 2>::vertex_handle, unsigned long>': return HandlePrinter(val)
  if str(val.type) == 'tatooine::pointset<double, 2>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::pointset<float, 2>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::pointset<double, 3>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::pointset<float, 3>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::pointset<double, 4>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::pointset<float, 4>::vertex_handle':  return HandlePrinter(val)

  if str(val.type) == 'tatooine::unstructured_simplicial_grid<double, 2>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_simplicial_grid<float, 2>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_simplicial_grid<double, 3>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_simplicial_grid<float, 3>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_simplicial_grid<double, 4>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_simplicial_grid<float, 4>::vertex_handle':  return HandlePrinter(val)

  if str(val.type) == 'tatooine::edgeset<double, 2>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::edgeset<float, 2>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::edgeset<double, 3>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::edgeset<float, 3>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::edgeset<double, 4>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::edgeset<float, 4>::vertex_handle':  return HandlePrinter(val)

  if str(val.type) == 'tatooine::unstructured_triangular_grid<double, 2>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_triangular_grid<float, 2>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_triangular_grid<double, 3>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_triangular_grid<float, 3>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_triangular_grid<double, 4>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_triangular_grid<float, 4>::vertex_handle':  return HandlePrinter(val)

  if str(val.type) == 'tatooine::unstructured_tetrahedral_grid<double, 2>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_tetrahedral_grid<float, 2>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_tetrahedral_grid<double, 3>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_tetrahedral_grid<float, 3>::vertex_handle':  return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_tetrahedral_grid<double, 4>::vertex_handle': return HandlePrinter(val)
  if str(val.type) == 'tatooine::unstructured_tetrahedral_grid<float, 4>::vertex_handle':  return HandlePrinter(val)
  
  return None

gdb.pretty_printers.append(tatooine_pretty_printers)
