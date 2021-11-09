%%
usos = [];
acum = [];
for i = 1:length(palstruct)
    tmp = palstruct(i).categorias;
    for j = 1:length(tmp)
        tmp(j).palabra = palstruct(i).palabra;
    end
    usos = [usos tmp];
    
    acum = [acum rmfield(palstruct(i),'categorias')];
end

struct2table(usos)
struct2table(acum)

writetable(usos,'usos.csv','Encoding','UTF-8')
writetable(acum,'acum.csv','Encoding','UTF-8')
