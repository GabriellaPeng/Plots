from Path_Test import root_path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

root_path = root_path + 'List1.xlsx'

df = pd.read_excel(root_path, sheet_name='Python', header=0,
                   names=["Main Client", "Industry", "Location", "Size"])

df["Size"] = pd.to_numeric(df["Size"])

colors = ['#ffe6cc', '#ffb366', '#ff8000', '#ff6666']
select = 'Location'
#industry, location = False, True

if select == 'Industry':
    df_ind = df.groupby(['Industry']).sum()
    values = df_ind.values.ravel()[::-1]
    colors = colors[:2]
    label = df['Industry'].value_counts().keys().to_list()
    explod = [0.1, 0.04]
    outer_label_size = 16
elif select == 'Location':
    df_loc = df.groupby(['Location']).sum()
    values = df_loc.values.ravel()
    colors = colors
    label = df['Location'].value_counts().keys().to_list()
    explod = [0.02, 0.02, 0.04, 0.06]
    outer_label_size =20

label_clients = df['Main Client'].values.tolist()
explod_client = [0.02 for i in range(len(df))]

fig, ax1 = plt.subplots(figsize=(10, 9))
sns.set_style()
# patche1, text1 = ax1.pie(values, colors=colors, labels=label,
#                          startangle=90, explode=explod, radius=2)

patches, texts = ax1.pie(df['Size'].ravel(), labels=df['Size'].ravel(),
                         explode=explod_client,  # autopct=[
                         # str(i) for i in df['Size'].ravel()]
                         startangle=90,
                         radius=1.6)
plt.axis('equal')
centre_circle = plt.Circle((0, 0), 1, fc='white', linewidth=0)
fig.gca().add_artist(centre_circle)
# plt.setp(text1, size=outer_label_size, weight="bold")
plt.setp(texts, weight="bold")
ax1.legend(patches, label_clients, loc='lower center',  ncol=len(label_clients))
plt.title(f'Current Main Clients Segmentation by \nSizes(Inner) & {select}(Outer)',
          fontsize=24)

print()
