from data_process import load_res_sim
from examples.load_data import load_valid_res

gofs = ['aic', 'mic', 'rmse', 'nse']  # 'rmse', 'nse', 'aic', 'mic'
algorithms = ['fscabc', 'mle', 'demcz', 'dream']  # 'fscabc', 'mle', 'demcz', 'dream'

type_res = 'top'  # 'top' , 'all'
type_res, type_sim = load_res_sim(type_res)

data = load_valid_res(algorithms, gofs, top=[True if type_sim ==
                                                              'top_weighted_sim' else
                                                              False][0], process_likes=True)