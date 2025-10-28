! mpi_hello_world.f90
! MPIとOpenMPIの基本的なテスト用プログラム (Fortran)

!mpif90 -o mpi_hello_world mpi_hello_world.f90
!mpirun -np 4 ./mpi_hello_world

program mpi_hello_world
  ! MPIモジュールをインポート
  use mpi

  ! 変数の宣言
  implicit none
  integer :: my_rank   ! 自身のプロセスID (ランク)
  integer :: num_procs ! 総プロセス数
  integer :: ierror    ! MPI関数のエラーコード

  ! MPI環境の初期化
  call MPI_Init(ierror)

  ! 自身のランク（ID）を取得
  call MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierror)

  ! 全プロセス数（サイズ）を取得
  call MPI_Comm_size(MPI_COMM_WORLD, num_procs, ierror)

  ! 挨拶メッセージを出力
  write(*,*) 'Hello from process ', my_rank, ' of ', num_procs

  ! MPI環境の終了処理
  call MPI_Finalize(ierror)

end program mpi_hello_world