import os


class SlurmJob(object):
    def __init__(self,
                 model_type,
                 python_file='/home/mila/q/qidong.yang/CP4GenerativeModel/main.py',
                 data_path='/home/mila/q/qidong.yang/CP4GenerativeModel/data/', 
                 experiment_root='', 
                 time='96:00:00', **kwargs):

        self.time = time
        self.kwargs = kwargs

        self.python_file = python_file
        self.data_path = data_path
        self.job_name = model_type + ''.join([f'--{k}={v}' for k, v in self.kwargs.items()])
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

        args = [f'--model_type {self.model_type}']
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

        part_1 = 'python -u '
        part_2 = self.python_file + self.args 
        part_3 = ' ' + f'--data_path={self.data_path}'
        part_4 = ' ' + f'--output_saving_path={self.output_path}'
        part_5 = ' ' + f'--model_path={self.model_path}'

        return part_1 + part_2 + part_3 + part_4 + part_5

    @property
    def setup(self):

        lines = [
            'singularity exec --overlay /scratch/qy707/torch_env/my_torch.ext3:ro ',
            '/scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif ',
            '/bin/bash -c "source /ext3/env.sh; ',
            self.command + '"'
        ]

        L = ''

        for line in lines:
            L = L + line

        return L

    @property
    def lines(self):

        lines = [
            '#!/bin/bash',
            f'#SBATCH --job-name={self.job_name}',
            f'#SBATCH --output={self.slurm_report_path + self.slurm_output_filename}',
            f'#SBATCH --error={self.slurm_report_path + self.slurm_error_filename}',
            '#SBATCH --nodes=1',
            f'#SBATCH --time={self.time}',
            '#SBATCH --cpus-per-task=4',
            '#SBATCH --mem=15Gb',
       #     f'#SBATCH --gres=gpu:rtx8000:1',
        ]

        lines = lines + ['', 'module purge', '', self.setup]

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

    model_type = 'flow-matching'
    experiment_root = f'/home/qy707/scratch/CP4Gen_Exp/{model_type}'
    python_file = '/home/qy707/CP4GenerativeModel/main.py'
    data_path = '/home/qy707/CP4GenerativeModel/data/'
    model_path = '/home/qy707/scratch/CP4Gen_Exp/models/'

    # datasets = ['s_curve']
    # datasets = ['s_curve', 'spiral', 'circle', 'moon', '25-Gaussians', '8-Gaussians']
    # epochs = [20000]
    
    job = SlurmJob(
                model_type=model_type, 
                experiment_root=experiment_root, 
                python_file=python_file, 
                data_path=data_path, 
                model_path=model_path, 
                dataset='s_curve', 
                n_epochs=20000, 
                n_ens=30)

    job.launch()


if __name__ == '__main__':

    synthetic_data_job()
