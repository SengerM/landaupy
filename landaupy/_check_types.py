def check_is_instance(obj, name: str, types):
	"""If `isinstance(obj, types)` is `True`, this function does nothing. 
	Otherwise, it rises a type error with a human friendly message.
	"""
	if isinstance(obj, types):
		return
	raise TypeError(f'`{name}` must be of type {repr(types)}, but received object of type {type(obj)}.')

def check_are_instances(objects_dict, types):
	"""Checks that all the objects in `objects_dict` are of one of the
	types in `types`. Otherwise, it rises a type error with a human 
	friendly message.
	
	Parameters
	----------
	objects_dict: dict
		Dictionary mapping objects names to the objects, for example
		`{'x': x, 'mu': mu, 'sigma': sigma}`
	types: a class name or a tuple of class names. 
		The types to which the objects should belong to. E.g. `float`,
		`(int, float, np.ndarray)`.
	"""
	for name, obj in objects_dict.items():
		check_is_instance(obj, name, types)
