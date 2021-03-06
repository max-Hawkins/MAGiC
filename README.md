# MAGiC: MeeKAT Analysis of GUPPI in C

## Usage

```
$ make
$ ./magic <GUPPI_file>
```


### Notes
The following programs can be useful when debugging GUPPI parsing.

> Display GUPPI file as raw bits and converted ASCII characters. Note: Each GUPPI header section is 80 bytes long - padded to that length with spaces after the header info.

```shell
$ xxd -b -l <bytes_to_read> -c <num_columns> -s <byte_start> <GUPPI_file>
```

> Time system calls in a C program with strace

```
$ strace -c ./magic <GUPPI_file>
```
