import os


# MODEL = ["ResUNet", "ResUNet5"]
CMD = f"""
python intInhom.py -N 4\\
                -sig 4\\
                -mu 1\\
                -nu 5\\
                -tol 1E-04 5E-05\\
                -dt 10\\
                -eps 1\\
                --sig_scl True 512 512\\
                --vismode False\\
                --visterm 0\\
                --dt_dir 'data/teeth/images'\\
                --range 41 50
        """
os.system(CMD)
