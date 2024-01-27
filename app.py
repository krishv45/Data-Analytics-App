
from flask import Flask, render_template, request, session
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import time
import random
import os
import json
import uuid




app = Flask(__name__)
app.secret_key = os.urandom(24)
df = pd.read_csv('world_population_data.csv')  # Replace 'world_population_data.csv' with your actual dataset file

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/plots')
def plots():
    return render_template('plots.html')


#Code for population density.........................................................................................................


@app.route('/population_density', methods=['GET', 'POST'])
def population_density():
    if request.method == 'POST':
        selected_countries = request.form.getlist('countries')
        selected_graph_types = request.form.getlist('graphType')

        if selected_countries and selected_graph_types:
            filtered_df = df[df['country'].isin(selected_countries)]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.set_style("whitegrid")
                    sns.barplot(x='country', y='population_density', data=filtered_df)
                    plt.title('Comparison of Population Density')
                    plt.xlabel('Country')
                    plt.ylabel('Population Density')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.set_style("darkgrid")
                    sns.lineplot(x='country', y='population_density', data=filtered_df, marker='o')
                    plt.title('Trend of Population Density')
                    plt.xlabel('Country')
                    plt.ylabel('Population Density')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    plt.figure(figsize=(5, 5))
                    plt.pie(filtered_df['population_density'], labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title('Distribution of Population Density')
                    plt.tight_layout()

                elif graph_type == 'scatter':
                    plt.figure(figsize=(7, 5))
                    sns.scatterplot(x='country', y='population_density', data=filtered_df, hue=filtered_df.country, marker='o', s=100)
                    plt.title('Scatter Plot of Population Density')
                    plt.xlabel('Country')
                    plt.ylabel('Population Density')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'histogram':
                    plt.figure(figsize=(7, 5))
                    sns.histplot(filtered_df['population_density'], bins=10, kde=True)
                    plt.title('Histogram of Population Density')
                    plt.xlabel('Population Density')
                    plt.ylabel('Frequency')
                    plt.tight_layout()


                timestamp = time.strftime("%Y%m%d-%H%M%S")
                random_identifier = str(uuid.uuid4().hex)

                chart_path = f'population_density_comparison_{timestamp}_{random_identifier}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('population_density.html', chart_paths=chart_paths, countries=df['country'].tolist())

    return render_template('population_density.html', countries=df['country'].tolist())




#Code for Growth rate..............................................................................................................


@app.route('/growth_rate', methods=['GET', 'POST'])
def growth_rate():
    if request.method == 'POST':
        selected_countries = request.form.getlist('countries')
        selected_graph_types = request.form.getlist('graphType')

        if selected_countries and selected_graph_types:
            filtered_df = df[df['country'].isin(selected_countries)]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(x='country', y='growth_rate', data=filtered_df)
                    plt.title('Comparison of Growth Rates')
                    plt.xlabel('Country')
                    plt.ylabel('Growth Rate')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(x='country', y='growth_rate', data=filtered_df)
                    plt.title('Trend of Growth Rates')
                    plt.xlabel('Country')
                    plt.ylabel('Growth Rate')
                    plt.xticks(rotation=45)
                    plt.tight_layout()


                elif graph_type == 'scatter':
                    plt.figure(figsize=(7, 5))
                    sns.scatterplot(x='country', y='population_density', data=filtered_df, hue=filtered_df.country, marker='o', s=100)
                    plt.title('Scatter Plot of Population Density')
                    plt.xlabel('Country')
                    plt.ylabel('Population Density')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'histogram':
                    plt.figure(figsize=(7, 5))
                    sns.histplot(filtered_df['population_density'], bins=10, kde=True)
                    plt.title('Histogram of Population Density')
                    plt.xlabel('Population Density')
                    plt.ylabel('Frequency')
                    plt.tight_layout()

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                random_identifier = str(uuid.uuid4().hex)

                chart_path = f'growth_rate_comparison_{timestamp}_{random_identifier}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('growth_rate.html', chart_paths=chart_paths, countries=df['country'].tolist())

    return render_template('growth_rate.html', countries=df['country'].tolist())

 


#Code for Area ......................................................................................................................




@app.route('/area', methods=['GET', 'POST'])
def area():
    if request.method == 'POST':
        selected_countries = request.form.getlist('countries')
        selected_graph_types = request.form.getlist('graphType')

        if selected_countries and selected_graph_types:
            filtered_df = df[df['country'].isin(selected_countries)]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(x='country', y='area', data=filtered_df)
                    plt.title('Comparison of Areas')
                    plt.xlabel('Country')
                    plt.ylabel('Area')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(x='country', y='area', data=filtered_df)
                    plt.title('Trend of Areas')
                    plt.xlabel('Country')
                    plt.ylabel('Area')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    plt.figure(figsize=(5, 5))
                    plt.pie(filtered_df['area'], labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title('Distribution of Areas')
                    plt.tight_layout()

                elif graph_type == 'scatter':
                    plt.figure(figsize=(7, 5))
                    sns.scatterplot(x='country', y='population_density', data=filtered_df, hue=filtered_df.country, marker='o', s=100)
                    plt.title('Scatter Plot of Population Density')
                    plt.xlabel('Country')
                    plt.ylabel('Population Density')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                elif graph_type == 'histogram':
                    plt.figure(figsize=(7, 5))
                    sns.histplot(filtered_df['population_density'], bins=10, kde=True)
                    plt.title('Histogram of Population Density')
                    plt.xlabel('Population Density')
                    plt.ylabel('Frequency')
                    plt.tight_layout()
                    
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                random_identifier = str(uuid.uuid4().hex)
                

                chart_path = f'area_comparison_{timestamp}_{random_identifier}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()




                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('area.html', chart_paths=chart_paths, countries=df['country'].tolist())

    return render_template('area.html', countries=df['country'].tolist())








#Code for 1980.............................................................................................................................

@app.route('/1980', methods=['GET', 'POST'])
def y1980():
    if request.method == 'POST':
        selected_countries = request.form.getlist('countries')
        selected_graph_types = request.form.getlist('graphType')

        if selected_countries and selected_graph_types:
            filtered_df = df[df['country'].isin(selected_countries)]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(x='country', y='1980', data=filtered_df)
                    plt.title('Population in 1980')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 1980')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(x='country', y='1980', data=filtered_df)
                    plt.title('Trend of Population in 1980')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 1980')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    plt.figure(figsize=(5, 5))
                    plt.pie(filtered_df['1980'], labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title('Distribution of Population in 1980')
                    plt.tight_layout()

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                random_identifier = str(uuid.uuid4().hex)

                chart_path = f'1980_comparison_{timestamp}_{random_identifier}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('1980.html', chart_paths=chart_paths, countries=df['country'].tolist())

    return render_template('1980.html', countries=df['country'].tolist())



#Code for 1990.....................................................................................................................


@app.route('/1990', methods=['GET', 'POST'])
def y1990():
    if request.method == 'POST':
        selected_countries = request.form.getlist('countries')
        selected_graph_types = request.form.getlist('graphType')

        if selected_countries and selected_graph_types:
            filtered_df = df[df['country'].isin(selected_countries)]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(x='country', y='1990', data=filtered_df)
                    plt.title('Population in 1990')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 1990')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(x='country', y='1990', data=filtered_df)
                    plt.title('Trend of Population in 1990')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 1990')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    plt.figure(figsize=(5, 5))
                    plt.pie(filtered_df['1990'], labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title('Distribution of Population in 1990')
                    plt.tight_layout()

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                random_identifier = str(uuid.uuid4().hex)

                chart_path = f'1990_comparison_{timestamp}_{random_identifier}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('1990.html', chart_paths=chart_paths, countries=df['country'].tolist())

    return render_template('1990.html', countries=df['country'].tolist())



#Code for 2000.........................................................................................................................

@app.route('/2000', methods=['GET', 'POST'])
def y2000():
    if request.method == 'POST':
        selected_countries = request.form.getlist('countries')
        selected_graph_types = request.form.getlist('graphType')

        if selected_countries and selected_graph_types:
            filtered_df = df[df['country'].isin(selected_countries)]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(x='country', y='2000', data=filtered_df)
                    plt.title('Population in 2000')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 2000')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(x='country', y='2000', data=filtered_df)
                    plt.title('Trend of Population in 2000')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 2000')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    plt.figure(figsize=(5, 5))
                    plt.pie(filtered_df['2000'], labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title('Distribution of Population in 2000')
                    plt.tight_layout()

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                random_identifier = str(uuid.uuid4().hex)

                chart_path = f'2000_comparison_{timestamp}_{random_identifier}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('2000.html', chart_paths=chart_paths, countries=df['country'].tolist())

    return render_template('2000.html', countries=df['country'].tolist())


#code for 2010...................................................................................................................

@app.route('/2010', methods=['GET', 'POST'])
def y2010():
    if request.method == 'POST':
        selected_countries = request.form.getlist('countries')
        selected_graph_types = request.form.getlist('graphType')

        if selected_countries and selected_graph_types:
            filtered_df = df[df['country'].isin(selected_countries)]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(x='country', y='2010', data=filtered_df)
                    plt.title('Population in 2010')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 2010')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(x='country', y='2010', data=filtered_df)
                    plt.title('Trend of Population in 2010')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 2010')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    plt.figure(figsize=(5, 5))
                    plt.pie(filtered_df['2010'], labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title('Distribution of Population in 2010')
                    plt.tight_layout()

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                random_identifier = str(uuid.uuid4().hex)

                chart_path = f'2010_comparison_{timestamp}_{random_identifier}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('2010.html', chart_paths=chart_paths, countries=df['country'].tolist())

    return render_template('2010.html', countries=df['country'].tolist())





    #code for 2020.....................................................................................................................

@app.route('/2020', methods=['GET', 'POST'])
def y2020():
    if request.method == 'POST':
        selected_countries = request.form.getlist('countries')
        selected_graph_types = request.form.getlist('graphType')

        if selected_countries and selected_graph_types:
            filtered_df = df[df['country'].isin(selected_countries)]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(x='country', y='2020', data=filtered_df)
                    plt.title('Population in 2020')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 2020')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(x='country', y='2020', data=filtered_df)
                    plt.title('Trend of Population in 2020')
                    plt.xlabel('Country')
                    plt.ylabel('Population in 2020')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    plt.figure(figsize=(5, 5))
                    plt.pie(filtered_df['2020'], labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title('Distribution of Population in 2020')
                    plt.tight_layout()

                timestamp = time.strftime("%Y%m%d-%H%M%S")
                random_identifier = str(uuid.uuid4().hex)

                chart_path = f'2020_comparison_{timestamp}_{random_identifier}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('2020.html', chart_paths=chart_paths, countries=df['country'].tolist())

    return render_template('2020.html', countries=df['country'].tolist())




#code for top largest countries and their density/growthrate....................................................................



@app.route('/largest_dens_gr', methods=['GET', 'POST'])
def largest_dens_gr():
    if request.method == 'POST':
        n_countries = int(request.form.get('n_countries'))
        selected_column = request.form.get('column')
        selected_graph_types = request.form.getlist('graph_types')

        if n_countries > 0 and selected_column and selected_graph_types:
            top_countries = df.nlargest(n_countries, 'area')

            filtered_df = top_countries[['country', selected_column]]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(x='country', y=selected_column, data=filtered_df)
                    plt.title(f'{selected_column.capitalize()} Comparison')
                    plt.xlabel('Country')
                    plt.ylabel(selected_column.capitalize())
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(x='country', y=selected_column, data=filtered_df)
                    plt.title(f'Trend of {selected_column.capitalize()}')
                    plt.xlabel('Country')
                    plt.ylabel(selected_column.capitalize())
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    values = filtered_df[selected_column]
                    if values.min() < 0:
                        # Handle negative values by converting them to zero
                        values = values.apply(lambda x: max(x, 0))

                    plt.figure(figsize=(5, 5))
                    plt.pie(values, labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title(f'Distribution of {selected_column.capitalize()}')
                    plt.tight_layout()

                chart_path = f'largest_dens_growth_{selected_column}_{graph_type}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': f'{graph_type.capitalize()} Chart', 'path': chart_path})

            return render_template('largest_dens_gr.html', chart_paths=chart_paths)

    return render_template('largest_dens_gr.html')



#code for top largest countries and their population....................................................................

@app.route('/largest_popln', methods=['GET', 'POST'])
def largest_popln():
    if request.method == 'POST':
        n = int(request.form.get('topN'))
        selected_years = request.form.getlist('years')
        selected_graph_types = request.form.getlist('graphType')

        if selected_years and selected_graph_types:
            filtered_df = df.nlargest(n, 'area')
            selected_columns = ['country'] + selected_years
            filtered_df = filtered_df[selected_columns]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(data=filtered_df.melt('country', var_name='year', value_name='population'), x='country', y='population', hue='year')
                    plt.title('Population Comparison by Country')
                    plt.xlabel('Country')
                    plt.ylabel('Population')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(data=filtered_df.melt('country', var_name='year', value_name='population'), x='country', y='population', hue='year')
                    plt.title('Population Trend by Country')
                    plt.xlabel('Country')
                    plt.ylabel('Population')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    plt.figure(figsize=(5, 5))
                    total_population = filtered_df[selected_years].sum(axis=1)
                    plt.pie(total_population, labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title('Population Distribution by Country')
                    plt.tight_layout()

                chart_path = f'population_graph_{graph_type}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('largest_popln.html', chart_paths=chart_paths)

    return render_template('largest_popln.html')



#code for top smallest countries and their density/growthrate....................................................................

@app.route('/smallest_dens_gr', methods=['GET', 'POST'])
def smallest_dens_gr():
    if request.method == 'POST':
        n_countries = int(request.form.get('n_countries'))
        selected_column = request.form.get('column')
        selected_graph_types = request.form.getlist('graph_types')

        if n_countries > 0 and selected_column and selected_graph_types:
            top_countries = df.nsmallest(n_countries, 'area')

            filtered_df = top_countries[['country', selected_column]]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(x='country', y=selected_column, data=filtered_df)
                    plt.title(f'{selected_column.capitalize()} Comparison')
                    plt.xlabel('Country')
                    plt.ylabel(selected_column.capitalize())
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(x='country', y=selected_column, data=filtered_df)
                    plt.title(f'Trend of {selected_column.capitalize()}')
                    plt.xlabel('Country')
                    plt.ylabel(selected_column.capitalize())
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    values = filtered_df[selected_column]
                    if values.min() < 0:
                        # Handle negative values by converting them to zero
                        values = values.apply(lambda x: max(x, 0))

                    plt.figure(figsize=(5, 5))
                    plt.pie(values, labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title(f'Distribution of {selected_column.capitalize()}')
                    plt.tight_layout()

                chart_path = f'smallest_dens_growth_{selected_column}_{graph_type}.png'

                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': f'{graph_type.capitalize()} Chart', 'path': chart_path})

            return render_template('smallest_dens_gr.html', chart_paths=chart_paths)

    return render_template('smallest_dens_gr.html')



#code for top smallest countries and their population....................................................................

@app.route('/smallest_popln', methods=['GET', 'POST'])
def smallest_popln():
    if request.method == 'POST':
        n = int(request.form.get('topN'))
        selected_years = request.form.getlist('years')
        selected_graph_types = request.form.getlist('graphType')

        if selected_years and selected_graph_types:
            filtered_df = df.nsmallest(n, 'area')
            selected_columns = ['country'] + selected_years
            filtered_df = filtered_df[selected_columns]

            chart_paths = []

            for graph_type in selected_graph_types:
                if graph_type == 'bar':
                    plt.figure(figsize=(7, 5))
                    sns.barplot(data=filtered_df.melt('country', var_name='year', value_name='population'), x='country', y='population', hue='year')
                    plt.title('Population Comparison by Country')
                    plt.xlabel('Country')
                    plt.ylabel('Population')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'line':
                    plt.figure(figsize=(7, 5))
                    sns.lineplot(data=filtered_df.melt('country', var_name='year', value_name='population'), x='country', y='population', hue='year')
                    plt.title('Population Trend by Country')
                    plt.xlabel('Country')
                    plt.ylabel('Population')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                elif graph_type == 'pie':
                    plt.figure(figsize=(5, 5))
                    total_population = filtered_df[selected_years].sum(axis=1)
                    plt.pie(total_population, labels=filtered_df['country'], autopct='%1.1f%%')
                    plt.title('Population Distribution by Country')
                    plt.tight_layout()

                chart_path = f'smallest_popln_{graph_type}.png'
                plt.savefig(f'static/{chart_path}')
                plt.close()

                chart_paths.append({'title': graph_type.capitalize() + ' Chart', 'path': chart_path})

            return render_template('smallest_popln.html', chart_paths=chart_paths)

    return render_template('smallest_popln.html')



#code for countries with highest population for 1980....2020 and their area, density, growthrate...........................


@app.route('/highest_pop_ar_den_gr', methods=['GET', 'POST'])
def highest_pop_ar_den_gr():
    if request.method == 'POST':
        selected_countries = request.form.getlist('countries')
        selected_columns = request.form.getlist('columns')
        selected_graph_types = request.form.getlist('graphType')

        if selected_countries and selected_columns and selected_graph_types:
            filtered_df = df[df['country'].isin(selected_countries)]

            chart_paths = []

            for column in selected_columns:
                for graph_type in selected_graph_types:
                    plt.figure(figsize=(7, 5))
                    
                    if graph_type == 'bar':
                        sns.barplot(x='country', y=column, data=filtered_df)
                    elif graph_type == 'line':
                        sns.lineplot(x='country', y=column, data=filtered_df)
                    elif graph_type == 'pie':
                        plt.pie(filtered_df[column], labels=filtered_df['country'], autopct='%1.1f%%')

                    plt.title(f'{column.capitalize()} - {graph_type.capitalize()} Chart')
                    plt.xlabel('Country')
                    plt.ylabel(column.capitalize())
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    random_identifier = str(uuid.uuid4().hex)

                    chart_path = f'{column}_{graph_type}_{timestamp}_{random_identifier}.png'
                    plt.savefig(f'static/{chart_path}')
                    plt.close()

                    chart_paths.append({'title': f'{graph_type.capitalize()} Chart', 'path': chart_path})

            return render_template('highest_pop_ar_den_gr.html', chart_paths=chart_paths, countries=df['country'].tolist())

    return render_template('highest_pop_ar_den_gr.html', countries=df['country'].tolist())





# Import the required libraries
import numpy as np




# Code for plotting graphs...................................................................................

@app.route('/plot_graphs', methods=['GET', 'POST'])
def plot_graphs():
    if request.method == 'POST':
        selected_graph_type = request.form.get('graphType')

        if selected_graph_type == 'boxplot':
            plt.figure(figsize=(7, 5))
            sns.boxplot(data=df)
            plt.title('Box Plot of Data')
            plt.xticks(rotation=45)
            plt.tight_layout()

        elif selected_graph_type == 'histogram':
            plt.figure(figsize=(7, 5))
            sns.histplot(data=df, bins=10, kde=True)
            plt.title('Histogram of Data')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.tight_layout()

        elif selected_graph_type == 'heatmap':
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title('Correlation Heatmap')
            plt.tight_layout()

        # Add more graph types as needed

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        random_identifier = str(uuid.uuid4().hex)

        chart_path = f'plot_{selected_graph_type}_{timestamp}_{random_identifier}.png'
        plt.savefig(f'static/{chart_path}')
        plt.close()

        return render_template('plot_graphs.html', chart_path=chart_path)

    return render_template('plot_graphs.html')



# Code for handling missing data.....................................................................................



@app.route('/handle_missing_data', methods=['GET', 'POST'])
def handle_missing_data():
    global df

    if request.method == 'POST':
        column_to_handle = request.form['column_to_handle']
        method = request.form['method']

        if column_to_handle == 'all':
            columns_with_missing_values = df.columns[df.isnull().any()]
            for column in columns_with_missing_values:
                handle_missing_values(df, column, method)
        else:
            handle_missing_values(df, column_to_handle, method)

        df.to_csv('world_population_data.csv', index=False)

    return render_template('handle_missing_data.html', columns=df.columns.tolist(), data=df.to_html())


def handle_missing_values(data_frame, column, method):
    if method == 'drop':
        data_frame.dropna(subset=[column], inplace=True)
    elif method == 'mean':
        data_frame[column].fillna(data_frame[column].mean(), inplace=True)
    elif method == 'median':
        data_frame[column].fillna(data_frame[column].median(), inplace=True)
    elif method == 'custom':
        custom_value = request.form['custom_value']
        data_frame[column].fillna(custom_value, inplace=True)
    elif method == 'drop_column':
        data_frame.drop(column, axis=1, inplace=True)






# ... (Previous code remains unchanged)

# New app route for previewing the dataset
@app.route('/preview_dataset')
def preview_dataset():
    num_rows_to_preview = None  # Display all rows

    preview_data = df.head(num_rows_to_preview)
    dataset_size = df.shape[0]  # Number of rows in the DataFrame
    num_columns = df.shape[1]   # Number of columns in the DataFrame

    return render_template('preview_dataset.html', data=preview_data.to_html(), dataset_size=dataset_size, num_columns=num_columns)



# ... Rename column.................................................................................

@app.route('/rename_columns', methods=['GET', 'POST'])
def rename_columns():
    global df  # Use the global keyword to work with the global df variable

    if request.method == 'POST':
        column_mapping = {}

        for column in df.columns:
            new_name = request.form.get(column)
            column_mapping[column] = new_name

        # Rename the columns based on the user-selected new names
        df.rename(columns=column_mapping, inplace=True)

        # Save the modified DataFrame back to the original CSV file, overwriting it
        df.to_csv('world_population_data.csv', index=False)

    else:  # If it's a GET request (page access or refresh), load DataFrame from CSV
        df = pd.read_csv('world_population_data.csv')

    return render_template('rename_columns.html', columns=df.columns.tolist(), data=df.to_html())


#Drop column...........................................................................................


@app.route('/drop_column', methods=['GET', 'POST'])
def drop_column():
    global df  # Use the global keyword to work with the global df variable

    if request.method == 'POST':
        column_to_drop = request.form['column_to_drop']

        # Drop the selected column from the DataFrame
        df.drop(column_to_drop, axis=1, inplace=True)

        # Save the modified DataFrame back to the original CSV file, overwriting it
        df.to_csv('world_population_data.csv', index=False)

    else:  # If it's a GET request (page access or refresh), load DataFrame from CSV
        df = pd.read_csv('world_population_data.csv')

    return render_template('drop_column.html', columns=df.columns.tolist(), data=df.to_html())


#outlier detection............................................................................

@app.route('/outlier_detection')
def outlier_detection():
    global df

    # Filter out columns containing non-numeric data (assuming all columns except 'country' are integer columns)
    numeric_columns = df.drop(columns='country').select_dtypes(include='number')

    # Perform outlier detection and generate the plot for each numeric column
    sns.set(style="whitegrid")
    plot_filepaths = []

    for column in numeric_columns.columns:
        # Using z-score to detect outliers in the current column
        z_scores = np.abs((numeric_columns[column] - numeric_columns[column].mean()) / numeric_columns[column].std())
        threshold = 3  # You can adjust the threshold based on your data
        outliers = df[z_scores > threshold]

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=column, data=numeric_columns)
        sns.scatterplot(x=column, y='country', data=outliers, color='red', label='Outliers')

        plt.xlabel(column)
        plt.title(f'Outlier Detection - {column}')
        plt.legend()
        plt.tight_layout()

        # Save the plot to a temporary file
        plot_filepath = f'static/temp_plot_{column}.png'
        plt.savefig(plot_filepath)
        plt.close()

        plot_filepaths.append(plot_filepath)

    return render_template('outlier_detection.html', columns=numeric_columns.columns.tolist(), plot_filepaths=plot_filepaths)





if __name__ == '__main__':
    app.run(debug=True)
