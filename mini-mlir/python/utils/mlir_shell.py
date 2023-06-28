import os
import subprocess
import logging


def _os_system(cmd: list, save_log: bool = False):
    cmd_str = ""
    for s in cmd:
        cmd_str += str(s) + " "
    if not save_log:
        print("[Running]: {}".format(cmd_str))
        ret = os.system(cmd_str)
        if ret == 0:
            print("[Success]: {}".format(cmd_str))
        else:
            raise RuntimeError("[!Error]: {}".format(cmd_str))
    else:
        _os_system_log(cmd_str)

def _os_system_log(cmd_str):
    # use subprocess to redirect the output stream
    # the file for saving the output stream should be set if using this function
    logging.info("[Running]: %s", cmd_str)

    process = subprocess.Popen(cmd_str,
                               shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)

    while True:
        output = process.stdout.readline().strip()
        if output == '' and process.poll() is not None:
            break
        if output:
            logging.info(output)

    process.wait()
    ret = process.returncode

    if ret == 0:
        logging.info("[Success]: %s", cmd_str)
    else:
        raise RuntimeError("[!Error]: {}".format(cmd_str))

# Model inference on CPU
def model_inference_cpu(objfile: str,
                        output_size: str):
    # generate executable file: a.out
    print("Generating executable file a.out ...")
    ccompiler = "clang"
    cname = "runtime_cpu.c"
    lib_name = ["libmlir_c_runner_utils.so.17git", "libmlir_c_runner_utils.so.17git", "libmlir_float16_utils.so.17git"]
    model = objfile
    lib_dir = os.getenv('PROJECT_ROOT')
    lib_list = [os.path.join(lib_dir,"capi/lib",name) for name in lib_name]
    cfile = os.path.join(lib_dir, "capi", cname)
    
    lib4 = "-lm"
    cflag = "-fPIC"
    cmd = [ccompiler, cfile, model, lib_list[0], lib_list[1], lib_list[2], lib4, cflag]
    _os_system(cmd)
    print("Successfully generate executable file a.out!")
    # execute model inference
    print("Runing ...")
    cmd1 = ["./a.out", output_size]
    _os_system(cmd1)
    print("Inference ends successfully! Results are saved in inference_result.txt.")

# Extra tool: delete file in current directory
def delete_file(file: str):
    cmd = ["rm -f", file]
    _os_system(cmd)
