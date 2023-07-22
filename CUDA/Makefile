
#########################################################################
#
# Makefile for cuda_test
#
#########################################################################

TEST_SOURCE = testCuda.cu

TARGETBIN := ./testCuda

CC = nvcc

$(TARGETBIN):$(TEST_SOURCE)
	$(CC)  $(TEST_SOURCE) -o $(TARGETBIN) -w
	-rm -rf *.exp *.lib

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN) *.exp *.lib
