INC_DIR = include
SRC_DIR = src
OBJ_DIR = build
LIB_DIR = lib
OBJSLIB = $(OBJ_DIR)/transpose.o\
		$(OBJ_DIR)/permutation_util.o\
		$(OBJ_DIR)/2dtranspose.o\
		$(OBJ_DIR)/3dtranspose.o\
		$(OBJ_DIR)/4dtranspose.o\
		$(OBJ_DIR)/5dtranspose.o\
		$(OBJ_DIR)/Nd_transpose.o\
		$(OBJ_DIR)/Nd_decompose.o\
		$(OBJ_DIR)/util.o\
		$(OBJ_DIR)/introspect.o\
		$(OBJ_DIR)/cudacheck.o\
		$(OBJ_DIR)/gcd.o\
		$(OBJ_DIR)/col_op.o\
		$(OBJ_DIR)/row_op.o\
		$(OBJ_DIR)/Nd_padding.o\
		$(OBJ_DIR)/reduced_math.o
OBJTEST = $(OBJ_DIR)/test_inplace.o\
		$(OBJ_DIR)/tensor_util.o
		
CC = g++
NVCC = nvcc
CUDAROOT = $(subst /bin/,,$(dir $(shell which $(NVCC))))
CPPFLAGS = -I$(INC_DIR) -I$(SRC_DIR) -I$(CUDAROOT)/include -L$(CUDAROOT)/lib64 -std=c++14 -O3 -DDEBUG
NVCCFLAGS = -gencode=arch=compute_86,code=sm_86 -rdc=true

all: lib/libinplacett.a test_inplace

test_inplace: lib/libinplacett.a $(OBJTEST)
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -L$(LIB_DIR) -linplacett -o $@ $^

lib/libinplacett.a: $(OBJSLIB)
	@mkdir -p $(LIB_DIR)
	rm -f lib/libinplacett.a
	ar -cvq lib/libinplacett.a $(OBJSLIB)
	
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(OBJ_DIR)/%.d
	$(NVCC) -c $(CPPFLAGS) $(NVCCFLAGS) -o $@ $<

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(OBJ_DIR)/%.d
	$(CC) -c $(CPPFLAGS) -o $@ $<
	
$(OBJ_DIR)/%.d: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	@echo "$(NVCC) -MM $(CPPFLAGS) $(NVCCFLAGS) $< > $@"
	@$(NVCC) -MM $(CPPFLAGS) $(NVCCFLAGS) $< > $@.tmp
	@sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.tmp > $@
	@rm -f $@.tmp

$(OBJ_DIR)/%.d: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	@echo "$(CC) -MM $(CPPFLAGS) $< > $@"
	@$(CC) -MM $(CPPFLAGS) $< > $@.tmp
	@sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.tmp > $@
	@rm -f $@.tmp
	
-include $(OBJSLIB:.o=.d)
	
clean:
	rm -f $(OBJ_DIR)/* lib/* $(EXEC)