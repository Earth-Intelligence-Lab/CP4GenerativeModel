import os


class SlurmJob(object):
    def __init__(self,
                 model_name,
                 python_file='/home/mila/q/qidong.yang/CP4GenerativeModel/main.py',
                 data_path='/home/mila/q/qidong.yang/CP4GenerativeModel/data/', 
                 experiment_root='', 
                 time='96:00:00', **kwargs):

        self.time = time
        self.kwargs = kwargs

        self.python_file = python_file
        self.data_path = data_path
        self.job_name = model_name + ''.join([f'--{k}={v}' for k, v in self.kwargs.items()])
        self.job_name = self.job_name.replace('(', '_')
        self.job_name = self.job_name.replace(')', '_')
        self.job_name = self.job_name.replace('[', '_')
        self.job_name = self.job_name.replace(']', '_')
        self.job_name = self.job_name.replace(' ', '_')
        self.job_name = self.job_name.replace(',', '_')
        self.job_name = self.job_name.replace('/', '_')
        self.job_name = self.job_name.replace('.', '_')

        self.experiment_path = experiment_root + '/' + self.job_name + '/'

        self.output_path = self.experiment_path + 'outputs' + '/'
        self.slurm_report_path = self.experiment_path + 'slurm_reports' + '/'
        self.slurm_code_path = self.experiment_path + 'slurm_codes' + '/'

        self.slurm_filename = 'slurm_script.sh'
        self.slurm_output_filename = 'slurm_output.txt'
        self.slurm_error_filename = 'slurm_error.txt'

    @property
    def args(self):

        args = []
        for k, v in self.kwargs.items():
            if isinstance(v, tuple):
                arg_str = f'--{k} ' + ' '.join([str(i) for i in v])
            else:
                arg_str = f'--{k} {v}'

            args.append(arg_str)

        if len(args) > 0:

            return ' ' + ' '.join(args)
        else:

            return ''

    @property
    def command(self):

        part_1 = 'python '
        part_2 = self.python_file + self.args + f' --data_path={self.data_path}'
        part_3 = ' ' + f'--output_saving_path={self.output_path}'

        return part_1 + part_2 + part_3

    @property
    def setup(self):

        lines = [
            'echo "starting job $SLURM_JOB_ID"',
            'module unload python',
            'module load anaconda/3',
            'conda activate obs_correction_torch_gpu',
        ]

        return lines

    @property
    def lines(self):

        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={self.job_name}',
            f'#SBATCH --output={self.slurm_report_path + self.slurm_output_filename}',
            f'#SBATCH --error={self.slurm_report_path + self.slurm_error_filename}',
            '#SBATCH --ntasks=1',
            f'#SBATCH --time={self.time}',
            '#SBATCH --cpus-per-task=4',
            '#SBATCH --mem=15Gb',
       #     f'#SBATCH --gres=gpu:rtx8000:1',
            '#SBATCH --partition=long',
        ]

        lines = lines + [''] + self.setup + ['', self.command]

        return lines

    @property
    def text(self):

        return '\n'.join(self.lines)

    def launch(self):

        os.system(f'mkdir -p {self.output_path}')
        os.system(f'mkdir -p {self.slurm_report_path}')
        os.system(f'mkdir -p {self.slurm_code_path}')

        with open(os.path.join(self.slurm_code_path, self.slurm_filename), 'w') as f:
            f.write(self.text)

        os.system(f'cat {os.path.join(self.slurm_code_path, self.slurm_filename)} | sbatch')


def synthetic_data_job():

    model_name = 'synthetic_data'
    experiment_root = f'/home/mila/q/qidong.yang/scratch/CP4Gen/{model_name}'

    # datasets = ['s_curve']
    datasets = ['s_curve', 'spiral', 'circle', 'moon', '25-Gaussians', '8-Gaussians']
    epochs = [2000, 5000, 10000, 20000, 50000]
    samples = [30]

    for dataset in datasets:
        for epoch in epochs:
            for sample in samples:
                job = SlurmJob(model_name=model_name, experiment_root=experiment_root, dataset=dataset, n_epochs=epoch, n_samples=sample)
                job.launch()


if __name__ == '__main__':

    synthetic_data_job()
